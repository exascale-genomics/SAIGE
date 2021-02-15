#ifndef SAIGE_LINALG_HPP
#define SAIGE_LINALG_HPP
#pragma once


#include <armadillo>


static inline arma::fmat xtx_gemm(const float alpha, const arma::fmat &x)
{
	const int M = x.n_rows;
	const int N = x.n_cols;
	arma::fmat ret(N, N);

	const char op_T = 'T';
	const char op_N = 'N';
	const int m = N;
	const int n = N;
	const int k = M;

	const int lda = M;
	const int ldb = M;
	float beta = 0.0;
	const int ldc = N;

	arma::blas::gemm(&op_T, &op_N, &m, &n, &k, &alpha, x.memptr(), &lda,
	x.memptr(), &ldb, &beta, ret.memptr(), &ldc);
	return ret;
}



static inline arma::fmat xtx_syrk(const float alpha, const arma::fmat &x)
{
  const int M = x.n_rows;
  const int N = x.n_cols;
  arma::fmat ret(N, N);
  
  // fill lower triangle with crossproducts
  const char tri_L = 'L';
  const char op_T = 'T';
  const float beta = 0.0f;
  arma::blas::syrk(&tri_L, &op_T, &N, &M, &alpha, x.memptr(), &M, &beta, ret.memptr(), &N);
  
  // copy lower triangle to upper
  float *ret_d = ret.memptr();
  const int blocksize = 8;
  for (int j=0; j<N; j+=blocksize)
  {
    for (int i=j+1; i<N; i+=blocksize)
    {
      for (int col=j; col<j+blocksize && col<N; ++col)
      {
        for (int row=i; row<i+blocksize && row<N; ++row)
          ret_d[col + N*row] = ret_d[row + N*col];
      }
    }
  }
  
  return ret;
}



static inline void mvp_gemv(const arma::fmat &A, const arma::fvec &x, arma::fvec &y)
{
  static const char trans = 'N';
  static const int m = A.n_rows;
  static const int n = A.n_cols;
  static const float alpha = 1.0f;
  static const float beta = 0.0f;
  static const int inc = 1;
  
  arma::blas::gemv(&trans, &m, &n, &alpha, A.memptr(), &m, x.memptr(), &inc, &beta, y.memptr(), &inc);
}


#endif
