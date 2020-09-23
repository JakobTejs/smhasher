// Based on Thorup's "high speed hashing for integers and strings"
// https://arxiv.org/pdf/1504.06804.pdf

#include <x86intrin.h>
#include <cstdint>
#include <cstring>

#ifndef tabulation_parallel_included
#define tabulation_parallel_included

static inline uint8_t  ttake08(const uint8_t *&p){ uint8_t  v; memcpy(&v, p, 1); p += 1; return v; }
static inline uint16_t ttake16(const uint8_t *&p){ uint16_t v; memcpy(&v, p, 2); p += 2; return v; }
static inline uint32_t ttake32(const uint8_t *&p){ uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
static inline uint64_t ttake64(const uint8_t *&p){ uint64_t v; memcpy(&v, p, 8); p += 8; return v; }

//const int TAB_STRIPES = 2;
//const int TAB_EXTRA = 2;

static uint64_t tab_parallel_rand64() {
   // we don't know how many bits we get from rand(),
   // but it is at least 16, so we concattenate a couple.
   uint64_t r = 0;
   for (int i = 0; i < 4; i++) {
      r <<= 16;
      r ^= rand();
   }
   return r;
}

static __uint128_t tab_parallel_rand128() {
   return ((__uint128_t)tab_parallel_rand64() << 64) | tab_parallel_rand64();
}


static const uint32_t TAB_PARALLEL_BLOCK_SIZE = 256;

static uint64_t tab_parallel_a;
static __uint128_t tab_parallel_b;
static __uint128_t tab_parallel_c;
static __uint128_t tab_parallel_d;
static uint64_t tab_parallel_random[TAB_PARALLEL_BLOCK_SIZE];
static uint64_t tab_parallel_random_2[TAB_PARALLEL_BLOCK_SIZE];
static __uint128_t tab_parallel_random_128[TAB_PARALLEL_BLOCK_SIZE];

static uint64_t tab_parallel_tabulation[8][256];

static uint64_t tab_parallel_combine(uint64_t acc, uint64_t block_hash, uint64_t a) {
   __uint128_t val = (__uint128_t)a * acc + block_hash;
   return (val & (((uint64_t)1 << 61) - 1)) + (val >> 61);// ^ (block_hash >> 64);
}


//# if defined(__AVX2__)
static uint64_t tab_parallel_one_block(const uint64_t* random, const uint8_t* data) {
   __m256i acc = _mm256_set_epi64x(0, 0, 0, 0);

   const  __m256i* const input  = (const __m256i *)data;
   const  __m256i* const _random = (const __m256i *)random;

   // We eat 256 bits (4 words) for each iteration
   for (size_t i = 0; i < TAB_PARALLEL_BLOCK_SIZE/4;) {
      __builtin_prefetch((data + 32*i + 384), 0 , 3 );

      for (size_t j = 0; j < 2; ++j, ++i) {
         __m256i const x = _mm256_loadu_si256(input + i);
         __m256i const a = _mm256_loadu_si256(_random + i);

         // Vector add x+a mod 2^32.
         // In contrast to mul, there is no epu version.
         __m256i const tmp  = _mm256_add_epi32(x, a);

         // Align high values into low values, to prepare for pair multiplication
         __m256i const tmp2 =  _mm256_srli_epi64(tmp, 32); //shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));

         // Multiplies the value of packed unsigned doubleword integer
         // in source vector s1 by the value in source vector s2 and stores
         // the result in the destination vector.
         // When a quadword result is too large to be represented in 64 bits
         // (overflow), the result is wrapped around and the low 64 bits are
         // written to the destination element (that is, the carry is ignored).
         __m256i const product =  _mm256_mul_epu32(tmp, tmp2);
         acc = _mm256_add_epi64(acc, product);
      }
   }
   uint64_t x[4];
   memcpy(&x, &acc, sizeof(x));
   // return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
   return (tab_parallel_d*(x[0] + x[1] + x[2] + x[3])) >> 64;
}

static uint64_t tab_parallel_stripe_block(const uint64_t* random, const uint8_t* data) {
   __m256i acc1 = _mm256_set_epi64x(0, 0, 0, 0);
   __m256i acc2 = _mm256_set_epi64x(0, 0, 0, 0);

   const  __m256i* const input  = (const __m256i *)data;
   const  __m256i* const _random = (const __m256i *)random;

   // We eat 256 bits (4 words) for each iteration
   for (size_t i = 0; i < TAB_PARALLEL_BLOCK_SIZE/4; i+=2) {
      __m256i const x1 = _mm256_loadu_si256(input + i);
      __m256i const x2 = _mm256_loadu_si256(input + i + 1);
      __m256i const a1 = _mm256_loadu_si256(_random + i);
      __m256i const a2 = _mm256_loadu_si256(_random + i + 1);

      __m256i tmp  = _mm256_add_epi32(x1, a1);
      __m256i tmp2 =  _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));
      __m256i product =  _mm256_mul_epu32(tmp, tmp2);
      acc1 = _mm256_add_epi64(acc1, product);

      tmp  = _mm256_add_epi32(x2, a2);
      tmp2 =  _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));
      product =  _mm256_mul_epu32(tmp, tmp2);
      acc2 = _mm256_add_epi64(acc2, product);
   }

   acc1 = _mm256_add_epi64(acc1, acc2);
   uint64_t x[4];
   memcpy(&x, &acc1, sizeof(x));
   return (tab_parallel_d*(x[0] + x[1] + x[2] + x[3])) >> 64;
}

static const __m128i tab_carryless_P64 = _mm_set_epi64x(0, (uint64_t)1 + ((uint64_t)1<<1) + ((uint64_t)1<<3) + ((uint64_t)1<<4));
static uint64_t tab_carryless_one_block(const uint64_t* random, const uint8_t* data) {
   __m128i acc = _mm_set_epi64x(0, 0);

   const  __m128i* const input  = (const __m128i *) data;
   const  __m128i* const _random = (const __m128i *)random;

   for (size_t i = 0; i < TAB_PARALLEL_BLOCK_SIZE/2; ++i) {
      __m128i x = _mm_loadu_si128(input + i);
      __m128i k = _mm_loadu_si128(_random + i);
      __m128i tmp = x ^ k;
      acc ^=  _mm_clmulepi64_si128(tmp, tmp, 0x10);
   }
   // Take the high bits for the first and low bits for the second
   __m128i q1 = _mm_clmulepi64_si128(acc, tab_carryless_P64, 0x01);
   // Take the high bits for the first and low bits for the second
   __m128i q2 = _mm_clmulepi64_si128(q2, tab_carryless_P64, 0x01);
   return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
}

// This gives 64 bit security
static uint64_t tab_parallel_double_block(
      const uint64_t* random1,
      const uint64_t* random2,
      const uint8_t* data) {
   __m256i acc = _mm256_set_epi64x(0, 0, 0, 0);

   const  __m256i* const input  = (const __m256i *)data;
   const  __m256i* const _random1 = (const __m256i *)random1;
   const  __m256i* const _random2 = (const __m256i *)random2;

   // We eat 256 bits (4 words) for each iteration
   for (size_t i = 0; i < TAB_PARALLEL_BLOCK_SIZE/4; ++i) {
      __m256i const x = _mm256_loadu_si256(input + i);
      __m256i const a1 = _mm256_loadu_si256(_random1 + i);
      __m256i const a2 = _mm256_loadu_si256(_random2 + i);

      __m256i tmp  = _mm256_add_epi32(x, a1);
      __m256i tmp2 =  _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));
      __m256i product =  _mm256_mul_epu32(tmp, tmp2);
      acc = _mm256_add_epi64(acc, product);

      tmp  = _mm256_add_epi32(x, a2);
      tmp2 =  _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));
      product =  _mm256_mul_epu32(tmp, tmp2);
      acc = _mm256_add_epi64(acc, product);
   }

   uint64_t x[4];
   memcpy(&x, &acc, sizeof(x));
   return (tab_parallel_d*(x[0] + x[1] + x[2] + x[3])) >> 64;
}
//# endif

static uint64_t tab_parallel_finalize_tabulation(uint64_t h) {
   uint8_t x[8];
   memcpy(&x, &h, sizeof(x));
   uint64_t res = 0;
   for (int i = 0; i < 8; i++)
      res ^= tab_parallel_tabulation[i][x[i]];
   return res;
}

static uint64_t tab_parallel_finalize_tabulation_32(uint32_t h) {
   uint8_t x[4];
   memcpy(&x, &h, sizeof(x));
   uint64_t res = 0;
   for (int i = 0; i < 4; i++)
      res ^= tab_parallel_tabulation[i][x[i]];
   return res;
}

static uint64_t tab_parallel_finalize_murmur(uint64_t h) {
   h ^= h >> 33;
   h *= 0xff51afd7ed558ccdULL;
   h ^= h >> 33;
   h *= 0xc4ceb9fe1a85ec53ULL;
   h ^= h >> 33;
   return h;
}

static uint64_t tab_parallel_finalize_poly(uint64_t h, int k) {
   uint64_t h0 = h >> 4;
   h = tab_parallel_random[0] % TAB_MERSENNE_61;
   for (int i = 1; i <= k; i++)
      h = tab_parallel_combine(h, h0, tab_parallel_random[i] % TAB_MERSENNE_61);
   if (h >= TAB_MERSENNE_61)
      h -= TAB_MERSENNE_61;
   return h;
}

static uint64_t tab_parallel_hash(const void* key, int len_bytes, uint32_t seed) {
   const uint8_t* data = (const uint8_t *)key;

   // We use len_bytes as the first character to distinguish vectors of
   // different length. We also add the seed, even though this hash function
   // really should be seeded using seed_init, rather than by passing it here.
   uint64_t val = (seed << 8) ^ len_bytes;

   if (len_bytes >= 8) {
      int len_words = len_bytes / 8;

      while (len_words >= TAB_PARALLEL_BLOCK_SIZE) {
         uint64_t block_hash = tab_parallel_one_block(tab_parallel_random, data);
        //  uint64_t block_hash = tab_carryless_one_block(tab_parallel_random, data);
         //uint64_t block_hash = tab_parallel_stripe_block(tab_parallel_random, data);
         //uint64_t block_hash = tab_parallel_double_block(tab_parallel_random, tab_parallel_random_2, data);
         val = tab_parallel_combine(val, block_hash >> 4, tab_parallel_a);

         // We eat TAB_PARALLEL_BLOCK_SIZE 8byte words.
         data += 8*TAB_PARALLEL_BLOCK_SIZE;
         len_words -= TAB_PARALLEL_BLOCK_SIZE;
      }

      // We want to use something simple here, that doesn't require warming
      // up AVX. We can use plain NH-hash or Mult-shift.
      for (size_t i = 0; i < len_words; ++i)
         val ^= (tab_parallel_random_128[i] * ttake64(data)) >> 64;

      // Even with MUM this seems slower
      // __uint128_t acc = 0;
      // for (size_t i = 0; i < len_words; ++i)
      //    acc += (__uint128_t)tab_parallel_random[i] * ttake64(data);
      // val ^= acc ^ (acc >> 64); // MUM
      //val ^= (tab_parallel_b * (uint64_t)acc) >> 64;
      //val ^= (tab_parallel_c * ((uint64_t)(acc >> 64))) >> 64;
   }

   uint32_t remaining_bytes = len_bytes % 8;
   if (remaining_bytes) {
      uint64_t last = 0;
      if (remaining_bytes & 4) last = ttake32(data);
      if (remaining_bytes & 2) last = (last << 16) | ttake16(data);
      if (remaining_bytes & 1) last = (last << 8) | ttake08(data);
      val ^= (tab_parallel_b * last) >> 64;
   }


   return tab_parallel_finalize_murmur(val);
   //return tab_parallel_finalize_tabulation(val);
   //return tab_parallel_finalize_tabulation_32((val >> 32) ^ (seed << 8) ^ len_bytes);
   //return tab_parallel_finalize_poly(val, 3);
}

static void tab_parallel_seed_init(uint32_t seed) {
   srand(seed);

   for (int i = 0; i < TAB_PARALLEL_BLOCK_SIZE; ++i) {
      tab_parallel_random[i] = tab_parallel_rand64();
      tab_parallel_random_2[i] = tab_parallel_rand64();
      tab_parallel_random_128[i] = tab_parallel_rand128();
   }
   tab_parallel_a = tab_parallel_rand64() >> 3;
   tab_parallel_b = tab_parallel_rand128();
   tab_parallel_c = tab_parallel_rand128();
   tab_parallel_d = tab_parallel_rand128();

   for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 256; ++j) {
         tab_parallel_tabulation[i][j] = tab_parallel_rand64();
      }
   }
}


#endif
