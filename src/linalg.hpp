#ifndef SAIGE_LINALG_HPP
#define SAIGE_LINALG_HPP
#pragma once


#include <armadillo>


namespace gpublas
{
  extern arma::fmat xtx_gemm(const float alpha, const arma::fmat &x, const int num_gpus=1);
  extern arma::fmat xtx_syrk(const float alpha, const arma::fmat &x, const int num_gpus=1);
}



namespace cpublas
{
  // copy lower triangle to upper
  static inline void symmetrize(arma::fmat &x)
  {
    const int n = x.n_rows;
    float *x_d = x.memptr();
    
    const int blocksize = 8;
    for (int j=0; j<n; j+=blocksize)
    {
      for (int i=j+1; i<n; i+=blocksize)
      {
        for (int col=j; col<j+blocksize && col<n; ++col)
        {
          for (int row=i; row<i+blocksize && row<n; ++row)
            x_d[col + n*row] = x_d[row + n*col];
        }
      }
    }
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
    
    symmetrize(ret);
    
    return ret;
  }
  
  
  
  static inline arma::fmat xtx_gemm(const float alpha, const arma::fmat &x)
  {
    const int m = x.n_rows;
    const int n = x.n_cols;
    arma::fmat ret(n, n);
    
    const char op_T = 'T';
    const char op_N = 'N';
    const float beta = 0.0f;
    
    arma::blas::gemm(&op_T, &op_N, &n, &n, &m, &alpha, x.memptr(), &m,
      x.memptr(), &m, &beta, ret.memptr(), &n);
    
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
}


#endif
