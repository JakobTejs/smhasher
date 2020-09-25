
#ifndef non_empty_test
#define non_empty_test

#include "Random.h"
#include "vmac.h"

#include <stdio.h>   // for printf
#include <memory.h>  // for memset
#include <math.h>    // for sqrt
#include <algorithm> // for sort, min
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// We view our timing values as a series of random variables V that has been
// contaminated with occasional outliers due to cache misses, thread
// preemption, etcetera. To filter out the outliers, we search for the largest
// subset of V such that all its values are within three standard deviations
// of the mean.

// double CalcMean ( std::vector<double> & v )
// {
//   double mean = 0;
  
//   for(int i = 0; i < (int)v.size(); i++)
//   {
//     mean += v[i];
//   }
  
//   mean /= double(v.size());
  
//   return mean;
// }

// double CalcMean ( std::vector<double> & v, int a, int b )
// {
//   double mean = 0;
  
//   for(int i = a; i <= b; i++)
//   {
//     mean += v[i];
//   }
  
//   mean /= (b-a+1);
  
//   return mean;
// }

// double CalcStdv ( std::vector<double> & v, int a, int b )
// {
//   double mean = CalcMean(v,a,b);

//   double stdv = 0;
  
//   for(int i = a; i <= b; i++)
//   {
//     double x = v[i] - mean;
    
//     stdv += x*x;
//   }
  
//   stdv = sqrt(stdv / (b-a+1));
  
//   return stdv;
// }

// double CalcStdv ( std::vector<double> & v )
// {
//   return CalcStdv(v, 0, v.size());
// }

//-----------------------------------------------------------------------------
// 256k blocks seem to give the best results.

uint32_t NonEmptyTest ( pfHash hash, uint32_t nr_elements, std::vector<uint8_t*>& elements, uint32_t blocksize, uint32_t nr_trial, Rand& r ) {
    // std::vector<uint8_t*> elements(nr_elements);
    // for (uint32_t i = 0; i < nr_elements; ++i) {
    //     elements[i] = new uint8_t[blocksize];
    //     r.rand_p(elements[i], blocksize);
    // }

    std::vector<bool> hit(nr_elements, false);
    uint32_t nr_non_empty = 0;
    for (uint8_t* element : elements) {
        uint64_t h;
        hash(element, blocksize, nr_trial, &h);
        h = h % nr_elements;
        if (!hit[h]) {
            hit[h] = true;
            ++nr_non_empty;
        }
    }

    return nr_non_empty;
}

const double EULER = 2.71828182845904523536;

void ShortNonEmptyTest ( pfHash hash, uint32_t seed )
{
  Rand r(seed);

  const int trials = 10;
  const int nr_elements = 10000000;
//   const int blocksize = 32 * 1024;

    // uint64_t last = r.rand_u64();
    // uint64_t a = r.rand_u64();
    // std::vector<uint8_t*> elements(nr_elements);
    // for (uint32_t i = 0; i < nr_elements; ++i) {
    //     last = last + a;
    //     elements[i] = new uint8_t[8];
    //     memcpy(elements[i], &last, 8);
    // }


  for(int i = 0; i < trials; i++)
  {
    uint64_t last = r.rand_u64();
    uint64_t a = r.rand_u64();
    std::vector<uint8_t*> elements(nr_elements);
    for (uint32_t i = 0; i < nr_elements; ++i) {
        last = last + a;
        elements[i] = new uint8_t[8];
        memcpy(elements[i], &last, 8);
    }
    Hash_Seed_init (hash, i);
    uint32_t nr_non_empty = NonEmptyTest(hash, nr_elements, elements, 8, i, r);

    printf("Number non-empty bins. Expected: %d Actual: %d\n", (int)((double)nr_elements*(1 - 1.0/EULER)), nr_non_empty);
    // hash(text,len,i,&hashes[i]);
  }

  fflush(NULL);
}

#endif
