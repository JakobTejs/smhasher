// Based on Thorup's "high speed hashing for integers and strings"
// https://arxiv.org/pdf/1504.06804.pdf

#include <x86intrin.h>
#include <cstdint>
#include <cstring>

#ifndef tabulation_parallel_included
#define tabulation_parallel_included

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
static const uint32_t TAB_PARALLEL_THRESHOLD = 128;

static uint64_t tab_parallel_a;
static __uint128_t tab_parallel_b;
static __uint128_t tab_parallel_c;
static __uint128_t tab_parallel_d;
static uint32_t tab_parallel_random[TAB_PARALLEL_BLOCK_SIZE];

static __uint128_t tab_parallel_short_b;
static __uint128_t tab_parallel_short_c;
static __uint128_t tab_parallel_short_random[TAB_PARALLEL_THRESHOLD];

static uint64_t tab_parallel_tabulation[8][256];

static const __m128i tab_parallel_P64 = _mm_set_epi64x(0, (uint64_t)1 + ((uint64_t)1<<1) + ((uint64_t)1<<3) + ((uint64_t)1<<4));

static uint64_t tab_parallel_combine(uint64_t acc, uint64_t block_hash, uint64_t a) {
    __uint128_t val = (__uint128_t)a * acc + block_hash;
    return (val & (((uint64_t)1 << 61) - 1)) + (val >> 61);// ^ (block_hash >> 64);
}


static uint64_t tab_parallel_one_block(const uint64_t* random, uint32_t seed, const char*& data) {
    __m256i acc = _mm256_set_epi64x(0, 0, 0, (uint64_t)seed << 32 | seed);

    
    const  __m256i* const input  = (const __m256i *) data;
    const  __m256i* const _random = (const __m256i *)random;

    for (size_t i = 0; i < TAB_PARALLEL_BLOCK_SIZE/4; ++i) {
        __m256i x = _mm256_loadu_si256(input + i);
        __m256i k = _mm256_loadu_si256(_random + i);
        // , k;
        // memcpy(&x, data, sizeof(x));
        // data += sizeof(x);

        // memcpy(&k, &random[i], sizeof(k));
        
        __m256i tmp  = _mm256_add_epi32(x, k);
        __m256i tmp2 =  _mm256_shuffle_epi32 (tmp, _MM_SHUFFLE(0, 3, 0, 1));
        
        __m256i product =  _mm256_mul_epu32(tmp, tmp2);
        acc = _mm256_add_epi64(acc, product);
	}
    uint64_t x[4];
    memcpy(&x, &acc, sizeof(x));
    // return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
    return (tab_parallel_d*(x[0] + x[1] + x[2] + x[3])) >> 64;
}

static uint64_t tab_parallel_almost_block(const uint64_t* random, uint32_t seed, const char*& data, uint32_t len) {
    __m128i acc = _mm_set_epi64x(0, (uint64_t)seed << 32 | seed);

    for (size_t i = 0; i < len/2; ++i) {
        __m128i x, k;
        memcpy(&x, data, sizeof(x));
        data += sizeof(x);

        memcpy(&k, &random[i << 1], sizeof(k));
        
        __m128i tmp = x ^ k;

        acc ^=  _mm_clmulepi64_si128(tmp, tmp, 0x10);
	}
    if (len & 1) {
        __uint64_t x;
        __m128i k;
        memcpy(&x, data, sizeof(x));
        data += 8;
        
        memcpy(&k, &random[len ^ 1], sizeof(k));

        acc ^=  _mm_clmulepi64_si128(_mm_set_epi64x(0, x), k, 0x00);
    }

   __m128i q1 = _mm_clmulepi64_si128(acc, tab_parallel_P64, 0x01); // Take the high bits for the first and low bits for the second
   __m128i q2 = _mm_clmulepi64_si128(q2, tab_parallel_P64, 0x01); // Take the high bits for the first and low bits for the second

    return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
}

static uint64_t tab_parallel_long(const uint64_t* random, uint32_t seed, const char* data, int len_bytes) {
    uint32_t len_words = len_bytes/8;

    uint64_t acc = 0;

    while (len_words >= TAB_PARALLEL_BLOCK_SIZE) {
        uint64_t block_hash = tab_parallel_one_block(random, seed, data);
        // data += 8*BLOCK_SIZE;
        len_words -= TAB_PARALLEL_BLOCK_SIZE;
        // _mm512_mul_epu32()
        acc = tab_parallel_combine(acc, block_hash >> 4, tab_parallel_a);
    }
    // if (len_words) {
    //     uint64_t block_hash = tab_parallel_almost_block(random, seed, data, len_words);

    //     acc = tab_parallel_combine(acc, block_hash, tab_parallel_a);
    // }

    // uint32_t remaining_bytes = len_bytes & ((1 << 8) - 1);
    // if (remaining_bytes) {
    //     uint64_t last = 0;
    //     if (remaining_bytes & 4) {
    //         uint32_t tmp;
    //         memcpy(&tmp, data, sizeof(tmp));
    //         last = tmp;
    //         data += 4;
    //     }
    //     if (remaining_bytes & 2) {
    //         uint16_t tmp;
    //         memcpy(&tmp, data, sizeof(tmp));
    //         last = (last << 16) | tmp;
    //         data += 2;
    //     }
    //     if (remaining_bytes & 1) {
    //         uint8_t tmp;
    //         memcpy(&tmp, data, sizeof(tmp));
    //         last = (last << 8) | tmp;
    //         data += 1;
    //     }

    //     acc ^= (tab_parallel_b * last) >> 64;
    // }

    // acc ^= (tab_parallel_c * len_bytes) >> 64;

    return acc;
}

static uint64_t tab_parallel_short(const uint64_t* random, uint32_t seed, const char* data, int len_bytes) {
    uint32_t len_words = len_bytes/8;

    uint64_t acc = (uint64_t)seed << 32 | seed;

    for (size_t i = 0; i < len_words; ++i) {
        uint64_t x;
        __uint128_t k;
        memcpy(&x, data, sizeof(x));
        data += sizeof(x);

        memcpy(&k, &random[i], sizeof(k));
        
        acc ^= (k * x) >> 64;
	}

    uint32_t remaining_bytes = len_bytes & ((1 << 8) - 1);
    if (remaining_bytes) {
        uint64_t last = 0;
        if (remaining_bytes & 4) {
            uint32_t tmp;
            memcpy(&tmp, data, sizeof(tmp));
            last = tmp;
            data += 4;
        }
        if (remaining_bytes & 2) {
            uint16_t tmp;
            memcpy(&tmp, data, sizeof(tmp));
            last = (last << 16) | tmp;
            data += 2;
        }
        if (remaining_bytes & 1) {
            uint8_t tmp;
            memcpy(&tmp, data, sizeof(tmp));
            last = (last << 8) | tmp;
            data += 1;
        }

        acc ^= (tab_parallel_short_b * last) >> 64;
    }

    acc ^= len_bytes;//(tab_parallel_short_c * len_bytes) >> 64;

    return acc;
}

static uint64_t tab_parallel_finalize_tabulation(uint64_t h) {
    uint8_t x[8];
    memcpy(&x, &h, sizeof(x));

    return tab_parallel_tabulation[0][x[0]] ^ tab_parallel_tabulation[1][x[1]]
         ^ tab_parallel_tabulation[2][x[2]] ^ tab_parallel_tabulation[3][x[3]]
         ^ tab_parallel_tabulation[4][x[4]] ^ tab_parallel_tabulation[5][x[5]]
         ^ tab_parallel_tabulation[6][x[6]] ^ tab_parallel_tabulation[7][x[7]];

}

static uint64_t tab_parallel_hash(const void* key, int len_bytes, uint32_t seed) {
    uint64_t val;
    if (len_bytes <= 8*TAB_PARALLEL_THRESHOLD) {
        val = tab_parallel_short((const uint64_t*)&tab_parallel_random, seed, (const char*) key, len_bytes);
    } else {
        val = tab_parallel_long((const uint64_t*)&tab_parallel_random, seed, (const char*) key, len_bytes);
    }
    return tab_parallel_finalize_tabulation(val);
}

static void tab_parallel_seed_init(uint32_t seed) {
    srand(seed);

    for (int i = 0; i < TAB_PARALLEL_BLOCK_SIZE; ++i) {
        tab_parallel_random[i] = tab_parallel_rand64();
    }
    tab_parallel_a = tab_parallel_rand64() >> 3;
    tab_parallel_b = tab_parallel_rand128();
    tab_parallel_c = tab_parallel_rand128();
    tab_parallel_d = tab_parallel_rand128();
   
    for (int i = 0; i < TAB_PARALLEL_THRESHOLD; ++i) {
       tab_parallel_short_random[i] = tab_parallel_rand128();
    }
    tab_parallel_short_b = tab_parallel_rand128();
    tab_parallel_short_c = tab_parallel_rand128();

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 256; ++j) {
            tab_parallel_tabulation[i][j] = tab_parallel_rand64();
        }
    }
}


#endif
