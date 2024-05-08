# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>

#include <algorithm>



#define N 169 
#define EPSILON 0.0001

void array_initialization();
void results_check();
void default_kernel();
void loop_tiling_kernel();
void openmp_kernel();
void results_check();
void vectorized_kernel();
void vectorized_kernel_register_blocking2();
void vectorized_kernel_register_blocking3();
void vectorized_kernel_register_blocking4();
void vectorized_kernel_register_blocking5();
void vectorized_kernel_register_blocking6();
void vectorized_kernel_register_blocking7();
void vectorized_kernel_register_blocking8();
void vectorized_kernel_register_blocking9();
void vectorized_kernel_register_blocking10();
void vectorized_kernel_register_blocking13();
unsigned short int equal(float const a, float const b);
float sum[N][N][N], A[N][N][N], C[N][N], sumOptimized[N][N][N];
//int  r, q, s, p;
bool b = false;


void main() {
      __int64 total_floating_point_operations = 2*N;
      total_floating_point_operations = total_floating_point_operations * N * N * N;
      double total_floating_point_operations_converted = total_floating_point_operations;
    array_initialization();
    double start = omp_get_wtime();
    default_kernel();
    double end = omp_get_wtime();
    double result = end - start;
    printf("Default kernel time: %lf\n", result);

   double flops = total_floating_point_operations_converted / 1000000;    
   flops = flops / result;
    printf("Default kernel megaFLOPS: %lf\n", flops);

    printf("\n");

    start = omp_get_wtime();
    loop_tiling_kernel();
    end = omp_get_wtime();
    result = end - start;
    printf("Loop tiling kernel time: %lf\n", result);
    results_check();

    //FLOPS calculation
    flops = total_floating_point_operations_converted / 1000000;
    flops = flops / result;
    printf("Loop tiling kernel megaFLOPS: %lf\n", flops);

    printf("\n");
      array_initialization();
     // s = 0;
     // p = 0;
      
      
      start = omp_get_wtime();
      openmp_kernel();
      end = omp_get_wtime();
      result = end - start;
      printf("OpenMP kernel time: %lf\n", result);
      results_check();

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
      printf("OpenMP kernel megaFLOPS: %lf\n", flops);
      
      printf("\n");

      array_initialization();
     // s = 0;
     // p = 0;
      
      start = omp_get_wtime();
      vectorized_kernel();
      end = omp_get_wtime();
      result = end - start;
      printf("Vectorized kernel time: %lf\n", result);
      results_check();

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* int Z = (N / 8) * 8;
      long remainder = N % 8;
      flops = (Z*16)*N*N*N;
      remainder = remainder * N * N * N;
      flops = flops + (2 * remainder);
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Vectorized kernel megaFLOPS: %lf\n", flops);
     
      printf("\n");

      array_initialization();
     // s = 0;
     // p = 0;
      
      start = omp_get_wtime();
      vectorized_kernel_register_blocking2();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(2): %lf\n", result);
     
      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* int P = (N / 8) * 8;
      int S = (N / 2) * 2;
      remainder = N % 8;
      long remainder_P = remainder * S * N * N;
      flops = (P * 32) * S * N * N;
      flops = flops + (4 * remainder_P);

      long remainder_S = N % 2;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(2) kernel megaFLOPS: %lf\n", flops);
      
      printf("\n");

      array_initialization();
     // s = 0;
    //  p = 0;
      
      start = omp_get_wtime();
      vectorized_kernel_register_blocking3();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(3): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
      /* P = (N / 8) * 8;
       S = (N / 3) * 3;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 48) * S * N * N;
      flops = flops + (6 * remainder_P);

      remainder_S = N % 3;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(3) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
 
      
      array_initialization();
     // s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking4();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(4): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* P = (N / 8) * 8;
      S = (N / 4) * 4;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 64) * S * N * N;
      flops = flops + (8 * remainder_P);

      remainder_S = N % 4;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(4) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
      
      array_initialization();
     // s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking5();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(5): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* P = (N / 8) * 8;
      S = (N / 5) * 5;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 80) * S * N * N;
      flops = flops + (10 * remainder_P);

      remainder_S = N % 5;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(5) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
      
      array_initialization();
      //s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking6();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(6): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* P = (N / 8) * 8;
      S = (N / 6) * 6;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 96) * S * N * N;
      flops = flops + (12 * remainder_P);

      remainder_S = N % 6;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(6) kernel megaFLOPS: %lf\n", flops);

      printf("\n");

      array_initialization();
     // s = 0;
     // p = 0;
      
      start = omp_get_wtime();
      vectorized_kernel_register_blocking7();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(7): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
      /*P = (N / 8) * 8;
      S = (N / 7) * 7;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 102) * S * N * N;
      flops = flops + (14 * remainder_P);

      remainder_S = N % 7;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(7) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
      
      array_initialization();
     // s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking8();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(8): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
    /*  P = (N / 8) * 8;
      S = (N / 8) * 8;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 118) * S * N * N;
      flops = flops + (16 * remainder_P);

      remainder_S = N % 8;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(8) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
      
      array_initialization();
     // s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking9();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(9): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* P = (N / 8) * 8;
      S = (N / 9) * 9;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 134) * S * N * N;
      flops = flops + (18 * remainder_P);

      remainder_S = N % 9;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(9) kernel megaFLOPS: %lf\n", flops);

      printf("\n");
      
      array_initialization();
     // s = 0;
     // p = 0;

      start = omp_get_wtime();
      vectorized_kernel_register_blocking10();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(10): %lf\n", result);

      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
      /*P = (N / 8) * 8;
      S = (N / 10) * 10;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 150) * S * N * N;
      flops = flops + (20 * remainder_P);

      remainder_S = N % 10;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(10) kernel megaFLOPS: %lf\n", flops);

      printf("\n");

      array_initialization();
     // s = 0;
     // p = 0;
      
      start = omp_get_wtime();
      vectorized_kernel_register_blocking13();
      end = omp_get_wtime();
      result = end - start;
      results_check();
      printf("Vectorized kernel with register blocking(13): %lf\n", result);
      
      //FLOPS calculation
      flops = total_floating_point_operations_converted / 1000000;
      flops = flops / result;
     /* P = (N / 8) * 8;
      S = (N / 13) * 13;
      remainder = N % 8;
      remainder_P = remainder * S * N * N;
      flops = (P * 208) * S * N * N;
      flops = flops + (26 * remainder_P);

      remainder_S = N % 13;
      remainder_S = remainder_S * P * 16;
      remainder_S = remainder_S + (remainder * 2);
      remainder_S = remainder_S * N * N;
      flops = flops / result;
      flops = flops / 1000000;*/
      printf("Register blocking(13) kernel megaFLOPS: %lf\n", flops);
       printf("Complete");
}




void array_initialization() {

    float e = 0.12, p = 0.72;
//    unsigned int i, j, k;
    if (b == false) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = (j % 9) + p;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    sum[i][j][k] = 0.0;
                    sumOptimized[i][j][k] = 0.0;
                    A[i][j][k] = (((i + j) % 99) + e);
                }
            }
        }
        b = true;
    }
    else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    sumOptimized[i][j][k] = 0.0;
                }

            }
        }
    }
}

void default_kernel() {

    for (int r = 0; r < N; r++) {
        for (int q = 0; q < N; q++) {
            for (int s = 0; s < N; s++) {
                for (int p = 0; p < N; p++) {
                    sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void openmp_kernel() {
#pragma omp parallel for
    for (int r = 0; r < N; r++) {
        for (int q = 0; q < N; q++) {
            for (int s = 0; s < N; s++) {
                for (int p = 0; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void loop_tiling_kernel() {
    int tile = 32;

    for (int ss = 0; ss < N; ss += tile) {
        for (int pp = 0; pp < N; pp += tile) {
            for (int r = 0; r < N; r++) {
                for (int q = 0; q < N; q++) {
                    for (int s = ss; s < std::min(ss + tile, N); s++) {
                        for (int p = pp; p < std::min(pp + tile, N); p++) {
                            sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                        }
                    }
                }
            }
        }
    }
    

}




void vectorized_kernel() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < N; s++) {
                __m256 AAVX = _mm256_set1_ps(A[r][q][s]); //
                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX, CAVX);
                    tmp = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], tmp);


                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking2() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 2) * 2); s += 2) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);

                for ( p = 0; p < (N / 8) * 8; p += 8) {

                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking3() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 3) * 3); s += 3) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);



                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];

                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking4() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 4) * 4); s += 4) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);

                for ( p = 0; p < (N / 8) * 8; p += 8) {

                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];

                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking5() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 5) * 5); s += 5) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);

                for ( p = 0; p < (N / 8) * 8; p += 8) {

                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];

                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking6() {
    int  r, q, s, p;

    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 6) * 6); s += 6) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);


                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking7() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 7) * 7); s += 7) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);
                __m256 AAVX7 = _mm256_set1_ps(A[r][q][s + 5]);


                for ( p = 0; p < (N / 8) * 8; p += 8) {

                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX7, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 6][p];

                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking8() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 8) * 8); s += 8) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);
                __m256 AAVX7 = _mm256_set1_ps(A[r][q][s + 6]);
                __m256 AAVX8 = _mm256_set1_ps(A[r][q][s + 7]);


                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX7, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX8, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 6][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 7][p];


                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking9() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 9) * 9); s += 9) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);
                __m256 AAVX7 = _mm256_set1_ps(A[r][q][s + 6]);
                __m256 AAVX8 = _mm256_set1_ps(A[r][q][s + 7]);
                __m256 AAVX9 = _mm256_set1_ps(A[r][q][s + 8]);


                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX7, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX8, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX9, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 6][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 7][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 8][p];
                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking10() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for (q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 10) * 10); s += 10) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);
                __m256 AAVX7 = _mm256_set1_ps(A[r][q][s + 6]);
                __m256 AAVX8 = _mm256_set1_ps(A[r][q][s + 7]);
                __m256 AAVX9 = _mm256_set1_ps(A[r][q][s + 8]);
                __m256 AAVX10 = _mm256_set1_ps(A[r][q][s + 9]);


                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX7, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX8, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX9, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX10, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 6][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 7][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 8][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 9][p];


                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}

void vectorized_kernel_register_blocking13() {
    int  r, q, s, p;
    for ( r = 0; r < N; r++) {
        for ( q = 0; q < N; q++) {
            for ( s = 0; s < ((N / 13) * 13); s += 13) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                __m256 AAVX2 = _mm256_set1_ps(A[r][q][s + 1]);
                __m256 AAVX3 = _mm256_set1_ps(A[r][q][s + 2]);
                __m256 AAVX4 = _mm256_set1_ps(A[r][q][s + 3]);
                __m256 AAVX5 = _mm256_set1_ps(A[r][q][s + 4]);
                __m256 AAVX6 = _mm256_set1_ps(A[r][q][s + 5]);
                __m256 AAVX7 = _mm256_set1_ps(A[r][q][s + 6]);
                __m256 AAVX8 = _mm256_set1_ps(A[r][q][s + 7]);
                __m256 AAVX9 = _mm256_set1_ps(A[r][q][s + 8]);
                __m256 AAVX10 = _mm256_set1_ps(A[r][q][s + 9]);
                __m256 AAVX11 = _mm256_set1_ps(A[r][q][s + 10]);
                __m256 AAVX12 = _mm256_set1_ps(A[r][q][s + 11]);
                __m256 AAVX13 = _mm256_set1_ps(A[r][q][s + 12]);

                for ( p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX2, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX3, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX4, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX5, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX6, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX7, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX8, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX9, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX10, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX11, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX12, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    tmp = _mm256_mul_ps(AAVX13, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);

                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 1][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 2][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 3][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 4][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 5][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 6][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 7][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 8][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 9][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 10][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 11][p];
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s + 1] * C[s + 12][p];


                }
            }

            for (; s < N; s++) {

                __m256 AAVX1 = _mm256_set1_ps(A[r][q][s]);
                for (p = 0; p < (N / 8) * 8; p += 8) {
                    __m256 CAVX = _mm256_loadu_ps(&C[s][p]);
                    __m256 sumAVX = _mm256_loadu_ps(&sumOptimized[r][q][p]);
                    __m256 tmp = _mm256_mul_ps(AAVX1, CAVX);
                    sumAVX = _mm256_add_ps(sumAVX, tmp);
                    _mm256_storeu_ps(&sumOptimized[r][q][p], sumAVX);

                }
                for (; p < N; p++) {
                    sumOptimized[r][q][p] = sumOptimized[r][q][p] + A[r][q][s] * C[s][p];
                }
            }
        }
    }
}


void results_check() {
    int  r, q, s, p;
    for (r = 0; r < N; r++) {
        for (q = 0; q < N; q++) {
            for (s = 0; s < N; s++) {
                for (p = 0; p < N; p++) {
                    if (equal(sum[r][q][p], sumOptimized[r][q][p]) == 1) {
                        printf("\n wrong values: %f %f", sum[r][q][p], sumOptimized[r][q][p]);
                    }

                }
            }
        }
    }
}

unsigned short int equal(float const a, float const b) {
    float temp = a - b;

    if (b == 0.0f) {
        if (a == 0.0f) {
            return 0;
        }
        else {
            return 1;
        }
    }
    else {

        if ((fabs(temp) / fabs(b)) < EPSILON) {
            return 0;
        }
        else {
            return 1;
        }
    }
}