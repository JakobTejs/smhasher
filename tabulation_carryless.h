// Based on Thorup's "high speed hashing for integers and strings"
// https://arxiv.org/pdf/1504.06804.pdf

#include <x86intrin.h>
#include <iostream>
// #include <cstdint>

#ifndef tabulation_carryless_included
#define tabulation_carryless_included

static uint64_t tab_carryless_rand64() {
   // we don't know how many bits we get from rand(),
   // but it is at least 16, so we concattenate a couple.
   uint64_t r = 0;
   for (int i = 0; i < 4; i++) {
      r <<= 16;
      r ^= rand();
   }
   return r;
}

static __uint128_t tab_carryless_rand128() {
   return ((__uint128_t)tab_carryless_rand64() << 64) | tab_carryless_rand64(); 
}


static const uint32_t TAB_CARRYLESS_BLOCK_SIZE = 256;
static const uint32_t TAB_CARRYLESS_THRESHOLD = 128;

static uint64_t tab_carryless_a;
static __uint128_t tab_carryless_b;
static __uint128_t tab_carryless_c;
static uint64_t tab_carryless_random[TAB_CARRYLESS_BLOCK_SIZE];

static __uint128_t tab_carryless_short_b;
static __uint128_t tab_carryless_short_c;
static __uint128_t tab_carryless_short_random[TAB_CARRYLESS_THRESHOLD];

static uint64_t tab_carryless_tabulation[8][256];

static const __m128i tab_carryless_P64 = _mm_set_epi64x(0, (uint64_t)1 + ((uint64_t)1<<1) + ((uint64_t)1<<3) + ((uint64_t)1<<4));

static uint64_t tab_carryless_combine(uint64_t acc, uint64_t block_hash, uint64_t a) {
    __uint128_t val = (__uint128_t)a * acc + block_hash;
    return (val & (((uint64_t)1 << 61) - 1)) + (val >> 61);// ^ (block_hash >> 64);
}


static uint64_t tab_carryless_one_block(const uint64_t* random, uint32_t seed, const char*& data) {
    __m128i acc = _mm_set_epi64x(0, (uint64_t)seed << 32 | seed);

    const  __m128i* const input  = (const __m128i *) data;
    const  __m128i* const _random = (const __m128i *)random;

    for (size_t i = 0; i < TAB_CARRYLESS_BLOCK_SIZE/2; ++i) {
        __m128i x = _mm_loadu_si128(input + i);
        __m128i k = _mm_loadu_si128(_random + i);
        
        __m128i tmp = x ^ k;

        acc ^=  _mm_clmulepi64_si128(tmp, tmp, 0x10);
	}
   __m128i q1 = _mm_clmulepi64_si128(acc, tab_carryless_P64, 0x01); // Take the high bits for the first and low bits for the second
   __m128i q2 = _mm_clmulepi64_si128(q2, tab_carryless_P64, 0x01); // Take the high bits for the first and low bits for the second

    return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
}

static uint64_t tab_carryless_almost_block(const uint64_t* random, uint32_t seed, const char*& data, uint32_t len) {
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

   __m128i q1 = _mm_clmulepi64_si128(acc, tab_carryless_P64, 0x01); // Take the high bits for the first and low bits for the second
   __m128i q2 = _mm_clmulepi64_si128(q2, tab_carryless_P64, 0x01); // Take the high bits for the first and low bits for the second

    return _mm_cvtsi128_si64(acc ^ q1 ^ q2);
}

static uint64_t tab_carryless_long(const uint64_t* random, uint32_t seed, const char* data, int len_bytes) {
    uint32_t len_words = len_bytes/8;

    uint64_t acc = 0;

    while (len_words >= TAB_CARRYLESS_BLOCK_SIZE) {
        uint64_t block_hash = tab_carryless_one_block(random, seed, data);
        // data += 8*BLOCK_SIZE;
        len_words -= TAB_CARRYLESS_BLOCK_SIZE;
        // _mm512_mul_epu32()
        acc = tab_carryless_combine(acc, block_hash >> 4, tab_carryless_a);
    }
    if (len_words) {
        uint64_t block_hash = tab_carryless_almost_block(random, seed, data, len_words);

        acc = tab_carryless_combine(acc, block_hash, tab_carryless_a);
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

        acc ^= (tab_carryless_b * last) >> 64;
    }

    acc ^= (tab_carryless_c * len_bytes) >> 64;

    return acc;
}

static uint64_t tab_carryless_short(const uint64_t* random, uint32_t seed, const char* data, int len_bytes) {
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

        acc ^= (tab_carryless_short_b * last) >> 64;
    }

    acc ^= len_bytes;//(tab_carryless_short_c * len_bytes) >> 64;

    return acc;
}

static uint64_t tab_carryless_finalize_tabulation(uint64_t h) {
    uint8_t x[8];
    memcpy(&x, &h, sizeof(x));

    return tab_carryless_tabulation[0][x[0]] ^ tab_carryless_tabulation[1][x[1]]
         ^ tab_carryless_tabulation[2][x[2]] ^ tab_carryless_tabulation[3][x[3]]
         ^ tab_carryless_tabulation[4][x[4]] ^ tab_carryless_tabulation[5][x[5]]
         ^ tab_carryless_tabulation[6][x[6]] ^ tab_carryless_tabulation[7][x[7]];

}

static uint64_t tab_carryless_hash(const void* key, int len_bytes, uint32_t seed) {
    uint64_t val;
    if (len_bytes <= 8*TAB_CARRYLESS_THRESHOLD) {
        val = tab_carryless_short((const uint64_t*)&tab_carryless_random, seed, (const char*) key, len_bytes);
    } else {
        val = tab_carryless_long((const uint64_t*)&tab_carryless_random, seed, (const char*) key, len_bytes);
    }
    return tab_carryless_finalize_tabulation(val);
}

static void tab_carryless_seed_init(uint32_t seed) {
    srand(seed);

    for (int i = 0; i < TAB_CARRYLESS_BLOCK_SIZE; ++i) {
        tab_carryless_random[i] = tab_carryless_rand64();
    }
    tab_carryless_a = tab_carryless_rand64() >> 3;
    tab_carryless_b = tab_carryless_rand128();
    tab_carryless_c = tab_carryless_rand128();
   
    for (int i = 0; i < TAB_CARRYLESS_THRESHOLD; ++i) {
       tab_carryless_short_random[i] = tab_carryless_rand128();
    }
    tab_carryless_short_b = tab_carryless_rand128();
    tab_carryless_short_c = tab_carryless_rand128();

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 256; ++j) {
            tab_carryless_tabulation[i][j] = tab_carryless_rand64();
        }
    }
}


#endif
