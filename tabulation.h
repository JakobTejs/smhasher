// Based on Thorup's "high speed hashing for integers and strings"
// https://arxiv.org/pdf/1504.06804.pdf

#include <x86intrin.h>
#include <cstdint>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////
// Private functions
////////////////////////////////////////////////////////////////////////////////

// Old take functions used by multiply-shift and poly
static inline uint8_t  take08(const uint8_t *p){ uint8_t  v; memcpy(&v, p, 1); return v; }
static inline uint16_t take16(const uint8_t *p){ uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t take32(const uint8_t *p){ uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t take64(const uint8_t *p){ uint64_t v; memcpy(&v, p, 8); return v; }

// New take functions, consuming the input
static inline uint8_t  ttake08(const uint8_t *&p){ uint8_t  v; memcpy(&v, p, 1); p += 1; return v; }
static inline uint16_t ttake16(const uint8_t *&p){ uint16_t v; memcpy(&v, p, 2); p += 2; return v; }
static inline uint32_t ttake32(const uint8_t *&p){ uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
static inline uint64_t ttake64(const uint8_t *&p){ uint64_t v; memcpy(&v, p, 8); p += 8; return v; }

namespace Tabulation {
   static const uint32_t BLOCK_SIZE = 256;
   static const uint64_t MERSENNE_61 = (1ull<<61) - 1;
   static const __uint128_t MERSENNE_127 = ((__uint128_t)1<<127) - 1;

   static uint64_t     a61;
   static __uint128_t  a127;
   static __uint128_t  b, c, d;
   static uint64_t     random[BLOCK_SIZE];
   static uint64_t     random_2[BLOCK_SIZE];
   static __uint128_t  random_128[BLOCK_SIZE];
   static uint64_t     table_64[8][256];
   static uint64_t     table_32[4][256];

   static const uint32_t MAXIMUM_REPETITION = 4;
   static uint64_t random_multiple[MAXIMUM_REPETITION][BLOCK_SIZE];

   static const __m128i P64 = _mm_set_epi64x(0, (uint64_t)1 + ((uint64_t)1<<1) + ((uint64_t)1<<3) + ((uint64_t)1<<4));

   static uint64_t combine61(uint64_t acc, uint64_t block_hash, uint64_t a) {
      __uint128_t val = (__uint128_t)a * acc + block_hash;
      return (val & ((1ull << 61) - 1)) + (val >> 61);
   }

   static __uint128_t combine127(__uint128_t acc, __uint128_t h, __uint128_t a) {
      // Compute acc * a + h  lazy mod 2^127-1
      uint64_t x[2], y[2];
      memcpy(&x, &a, sizeof(x));
      memcpy(&y, &acc, sizeof(y));
      // I don't actually know if I've flipped the high and low bits here
      __uint128_t y0x0 = (__uint128_t)y[0]*x[0];
      __uint128_t y0x1 = (__uint128_t)y[0]*x[1];
      __uint128_t y1x0 = (__uint128_t)y[1]*x[0];
      __uint128_t y1x1 = (__uint128_t)y[1]*x[1];
      // Compute (y[0]*2^64 + y[1])(x[0]*2^64 + x[1])
      //       = y[0]x[0]*2^128 + (y[0]x[1] + y[1]x[0])*2^64 + y[1]x[1]
      // Shifted down 127: (Separate shifts to avoid overflow problems.)
      // Note that we actually assume (h >> 127) is 0.
      __uint128_t hi = (y0x0 << 1) + (y0x1 >> 63) + (y1x0 >> 63) + (y1x1 >> 127) + (h >> 127);
      // And the low 127 bits: (Overflows are less of an issue here.)
      __uint128_t lo = (((y0x1 + y1x0) << 64) + y1x1 + h) & MERSENNE_127;
      return hi + lo;
   }

   static __uint128_t rand128() {
      // we don't know how many bits we get from rand(),
      // but it is at least 16, so we concattenate a couple.
      __uint128_t r = 0;
      for (int i = 0; i < 8; i++) {
         r <<= 16;
         r ^= rand();
      }
      return r;
   }

   static void seed_init(uint32_t seed) {
      srand(seed);

      for (int i = 0; i < BLOCK_SIZE; ++i) {
         random[i] =     rand128();
         random_2[i] =   rand128();
         random_128[i] = rand128();
         for (int j = 0; j < Tabulation::MAXIMUM_REPETITION; ++j)
            random_multiple[j][i] = rand128();
      }
      a61 = rand128() % MERSENNE_61;
      a127 = rand128() % MERSENNE_127;
      b = rand128();
      c = rand128();
      d = rand128();

      for (int j = 0; j < 256; ++j) {
         for (int i = 0; i < 8; i++)
            table_64[i][j] = rand128();
         for (int i = 0; i < 4; i++)
            table_32[i][j] = rand128();
      }
   }

   namespace Block {
      #if defined(__SSE4_2__) && defined(__x86_64__)
      static uint64_t vector_nh32(const uint64_t* random, const uint8_t* data) {
         __m256i acc = _mm256_set_epi64x(0, 0, 0, 0);

         const  __m256i* const input  = (const __m256i *)data;
         const  __m256i* const _random = (const __m256i *)random;

         // We eat 256 bits (4 words) for each iteration
         for (size_t i = 0; i < BLOCK_SIZE/4;) {
            __builtin_prefetch((data + 32*i + 384), 0 , 3 );

            for (size_t j = 0; j < 2; ++j, ++i) {
               __m256i const x = _mm256_loadu_si256(input + i);
               __m256i const a = _mm256_loadu_si256(_random + i);

               // Vector add x+a mod 2^32.
               // In contrast to mul, there is no epu version.
               // We actually add in 64 bit chunks, for consistency with
               // non-avx version.
               __m256i const tmp  = _mm256_add_epi64(x, a);
               //__m256i const tmp  = _mm256_add_epi32(x, a);

               // Align high values into low values, to prepare for pair multiplication
               __m256i const tmp2 = _mm256_srli_epi64(tmp, 32);
               // An alternative using shuffeling
               //__m256i const tmp2 = shuffle_epi32(tmp, _MM_SHUFFLE(0, 3, 0, 1));

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
         return x[0] + x[1] + x[2] + x[3];
      }

      static __uint128_t carryless(const uint64_t* random, const uint8_t* data) {
         __m128i acc = _mm_set_epi64x(0, 0);

         const  __m128i* const input  = (const __m128i *) data;
         const  __m128i* const _random = (const __m128i *)random;

         for (size_t i = 0; i < BLOCK_SIZE/2; ) {
            __builtin_prefetch((data + 16*i + 384), 0 , 3 );

            for (size_t j = 0; j < 4; ++j, ++i) {
               __m128i x = _mm_loadu_si128(input + i);
               __m128i k = _mm_loadu_si128(_random + i);
               __m128i tmp = x ^ k;
               acc ^=  _mm_clmulepi64_si128(tmp, tmp, 0x10);
            }
         }

         // We don't have to complete the modulo. Just return all 128 bits.
         __uint128_t res;
         memcpy(&res, &acc, 16);
         return res;

         // Complete the modulo P64.
         // Take the high bits for the first and low bits for the second
         // __m128i q1 = _mm_clmulepi64_si128(acc, P64, 0x01);
         // Take the high bits for the first and low bits for the second
         // __m128i q2 = _mm_clmulepi64_si128(q2, P64, 0x01);
         // return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
      }

      // This gives 64 bit security by using mul_epu32 twice.
      // Seems to be slower than 64 bit carry-less on our machines.
      static uint64_t vector_nh32_double(
            const uint64_t* random1,
            const uint64_t* random2,
            const uint8_t* data) {
         __m256i acc = _mm256_set_epi64x(0, 0, 0, 0);

         const  __m256i* const input  = (const __m256i *)data;
         const  __m256i* const _random1 = (const __m256i *)random1;
         const  __m256i* const _random2 = (const __m256i *)random2;

         // We eat 256 bits (4 words) for each iteration
         for (size_t i = 0; i < BLOCK_SIZE/4;) {
            __builtin_prefetch((data + 16*i + 384), 0 , 3 );
            for(size_t j = 0; j < 8; ++j, ++i) {
               __m256i const x = _mm256_loadu_si256(input + i);
               __m256i const a1 = _mm256_loadu_si256(_random1 + i);
               __m256i const a2 = _mm256_loadu_si256(_random2 + i);

               __m256i tmp  = _mm256_add_epi32(x, a1);
               __m256i tmp2 = _mm256_srli_epi64(tmp, 32);
               __m256i product =  _mm256_mul_epu32(tmp, tmp2);
               acc = _mm256_add_epi64(acc, product);

               tmp  = _mm256_add_epi32(x, a2);
               tmp2 = _mm256_srli_epi64(tmp, 32);
               product =  _mm256_mul_epu32(tmp, tmp2);
               acc = _mm256_add_epi64(acc, product);
            }
         }
      }

      template<const uint32_t REPS>
      struct nh32_multiple_return {
         uint64_t vals[REPS];
      };

      template<const uint32_t REPS>
      static nh32_multiple_return<REPS> vector_nh32_multiple(const uint64_t* random_multiple[REPS],  const uint8_t* data) {
         __m256i acc[REPS]; 

         const  __m256i* const input  = (const __m256i *)data;
         const  __m256i* const _random[REPS]  = (const __m256i *)random_multiple;

         // We eat 256 bits (4 words) for each iteration
         for (size_t i = 0; i < BLOCK_SIZE/4; ++i) {
            //  __builtin_prefetch((data + 16*i + 384), 0 , 3 );
            //  for(size_t j = 0; j < 2; ++j, ++i) {
            __m256i const x = _mm256_loadu_si256(input + i);

            __m256i a[REPS];
            for (size_t l = 0; l < REPS; ++l) {
               a[l]            = _mm256_loadu_si256(_random[l] + i);
               __m256i tmp     = _mm256_add_epi32(x, a[l]);
               __m256i tmp2    = _mm256_srli_epi64(tmp, 32);
               __m256i product = _mm256_mul_epu32(tmp, tmp2);
               acc[l]          = _mm256_add_epi64(acc[l], product);
            }
            //  }
         }

         uint64_t vals[REPS];
         for (size_t l = 0; l < REPS; ++l) {
            uint64_t x[4];
            memcpy(&x, acc + l, sizeof(x));
            vals[l] = x[0] + x[1] + x[2] + x[3];
         }

         nh32_multiple_return<REPS> ret;
         memcpy(&ret, vals, sizeof(ret));

         return ret;
      }
      #endif

      static uint64_t scalar_nh32(const uint64_t* random, const uint8_t* data) {
         uint64_t block_hash = 0;
         for (size_t i = 0; i < BLOCK_SIZE; i++) {
            // Parallel addition helps auto-vectorization
            uint64_t x = random[i] + ttake64(data);
            block_hash += (x >> 32)*(uint32_t)x;
         }
         return block_hash;
      }

      static __uint128_t scalar_nh64(const uint64_t* random, const uint8_t* data) {
         __uint128_t block_hash = 0;
         for (size_t i = 0; i < BLOCK_SIZE; i+=2) {
            uint64_t x1 = random[i] + ttake64(data);
            uint64_t x2 = random[i+1] + ttake64(data);
            block_hash += (__uint128_t)x1*x2;
         }
         return block_hash;
      }
   }

   namespace Finalizer {
      static uint64_t tabulation_64(uint64_t h) {
         uint8_t x[8];
         memcpy(&x, &h, sizeof(x));
         uint64_t res = 0;
         for (int i = 0; i < 8; i++)
            res ^= Tabulation::table_64[i][x[i]];
         return res;
      }

      static uint32_t tabulation_32(uint32_t h) {
         uint8_t x[4];
         memcpy(&x, &h, sizeof(x));
         uint32_t res = 0;
         for (int i = 0; i < 4; i++)
            res ^= Tabulation::table_32[i][x[i]];
         return res;
      }

      static uint64_t murmur(uint64_t h) {
         h ^= h >> 33;
         h *= 0xff51afd7ed558ccdULL;
         h ^= h >> 33;
         h *= 0xc4ceb9fe1a85ec53ULL;
         h ^= h >> 33;
         return h;
      }

      static uint64_t poly(uint64_t h, int k) {
         uint64_t h0 = h >> 4;
         h = random[0] % MERSENNE_61;
         for (int i = 1; i <= k; i++)
            h = combine61(h, h0, random[i] % MERSENNE_61);
         if (h >= MERSENNE_61)
            h -= MERSENNE_61;
         return h;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////

static void tabulation_seed_init(uint32_t seed) {
   Tabulation::seed_init(seed);
}

static uint64_t tabulation_64_hash(const void* key, int len_bytes, uint32_t seed) {
   const uint8_t* data = (const uint8_t *)key;

   // We use len_bytes as the first character to distinguish vectors of
   // different length. We also add the seed, even though this hash function
   // really should be seeded using seed_init, rather than by passing it here.
   uint64_t val = len_bytes;

   if (len_bytes >= 8) {
      int len_words = len_bytes / 8;

      if (len_words >= Tabulation::BLOCK_SIZE) {
         // To keep 64 bit internal state, we need a prime larger than 2^64.
         // Hence the value needs to be stored in 128 bits.
         __uint128_t val127 = val;
         while (len_words >= Tabulation::BLOCK_SIZE) {
            #if defined(__SSE4_2__) && defined(__x86_64__)
               __uint128_t block_hash = Tabulation::Block::carryless(Tabulation::random, data);
            #elif
               __uint128_t block_hash = Tabulation::Block::scalar_nh64(Tabulation::random, data);
            #endif

            // Split block_hash into two 64 bit characters that fit under Mersenne-127.
            uint64_t x[2];
            memcpy(&x, &block_hash, sizeof(x));
            // Add both characters to the variable length hash.
            val127 = Tabulation::combine127(val127, x[0], Tabulation::a127);
            val127 = Tabulation::combine127(val127, x[1], Tabulation::a127);

            // We eat TAB_PARALLEL_BLOCK_SIZE 8-byte words. The block-hashes don't
            // automatically increment data.
            data += 8*Tabulation::BLOCK_SIZE;
            len_words -= Tabulation::BLOCK_SIZE;
         }
         // Now reduce back down to 64 bits.
         uint64_t x[2];
         memcpy(&x, &val127, sizeof(x));
         val = (Tabulation::b + x[0])*(Tabulation::c + x[1]) >> 64;
      }

      // We want to use something simple here, that doesn't require warming
      // up AVX. We can use plain NH-hash or Mult-shift.
      for (size_t i = 0; i < len_words; ++i)
         val += Tabulation::random_128[i] * ttake64(data) >> 64;
   }

   int remaining_bytes = len_bytes % 8;
   if (remaining_bytes) {
      uint64_t last = 0;
      if (remaining_bytes & 4) last = ttake32(data);
      if (remaining_bytes & 2) last = (last << 16) | ttake16(data);
      if (remaining_bytes & 1) last = (last << 8) | ttake08(data);
      val += (Tabulation::b * last) >> 64;
   }


   return Tabulation::Finalizer::tabulation_64(val);
}

static uint64_t tabulation_32_hash(const void* key, int len_bytes, uint32_t seed) {
   const uint8_t* data = (const uint8_t *)key;

   uint64_t val = len_bytes;

   if (len_bytes >= 8) {
      int len_words = len_bytes / 8;

      while (len_words >= Tabulation::BLOCK_SIZE) {
         #if defined(__AVX2__) || defined(__SSE4_2__) && defined(__x86_64__)
            uint64_t block_hash = Tabulation::Block::vector_nh32(Tabulation::random, data);
         #else
            uint64_t block_hash = Tabulation::Block::scalar_nh32(Tabulation::random, data);
         #endif

         // Add to variable-length hash as two characters to fit under Mersenne 61.
         // Note that the "bottle-neck" is still 32 bits.
         uint32_t x[2];
         memcpy(&x, &block_hash, sizeof(x));
         val = Tabulation::combine61(val, x[0], Tabulation::a61);
         val = Tabulation::combine61(val, x[1], Tabulation::a61);

         // The block hashes don't automatically increment data
         data += 8*Tabulation::BLOCK_SIZE;
         len_words -= Tabulation::BLOCK_SIZE;
      }

      for (size_t i = 0; i < len_words; ++i)
         val += Tabulation::random_128[i] * ttake64(data) >> 64;
   }

   int remaining_bytes = len_bytes % 8;
   if (remaining_bytes) {
      uint64_t last = 0;
      if (remaining_bytes & 4) last = ttake32(data);
      if (remaining_bytes & 2) last = (last << 16) | ttake16(data);
      if (remaining_bytes & 1) last = (last << 8) | ttake08(data);
      val += (Tabulation::b * last) >> 64;
   }

   // We use len_bytes as a character to distinguish vectors of
   // different length. We also add the seed, even though this hash function
   // really should be seeded using seed_init, rather than by passing it here.
   //return Tabulation::Finalizer::tabulation_32((val >> 32) ^ (seed << 8) ^ len_bytes);
   //return Tabulation::Finalizer::tabulation_64((val >> 32) ^ (seed << 8) ^ len_bytes);

   // We can't just reduce val to 32 bits, since it may not be the
   // result of a multiply-shift hash.
   return Tabulation::Finalizer::tabulation_64(val);
}

