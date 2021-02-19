#include <armadillo>
#include <cstdlib>
#include <cublasXt.h>
#include <omp.h>
#include <stdexcept>

#include "linalg.hpp"


namespace gpublas
{
  static const int blockdim = 2048;
  static const float cpu_ratio = 0.0f;
  
  
  
  namespace err
  {
    inline std::string get_cublas_error_msg(cublasStatus_t check)
    {
      if (check == CUBLAS_STATUS_SUCCESS)
        return "";
      else if (check == CUBLAS_STATUS_NOT_INITIALIZED)
        return "cuBLAS not initialized";
      else if (check == CUBLAS_STATUS_ALLOC_FAILED)
        return "internal cuBLAS memory allocation failed";
      else if (check == CUBLAS_STATUS_INVALID_VALUE)
        return "unsupported parameter";
      else if (check == CUBLAS_STATUS_ARCH_MISMATCH)
        return "function requires feature missing from device architecture";
      else if (check == CUBLAS_STATUS_MAPPING_ERROR)
        return "access to GPU memory space failed";
      else if (check == CUBLAS_STATUS_EXECUTION_FAILED)
        return "GPU program failed to execute";
      else if (check == CUBLAS_STATUS_INTERNAL_ERROR)
        return "internal cuBLAS operation failed";
      else if (check == CUBLAS_STATUS_NOT_SUPPORTED)
        return "requested functionality is not supported";
      else if (check == CUBLAS_STATUS_LICENSE_ERROR)
        return "error with cuBLAS license check";
      else
        return "unknown cuBLAS error occurred";
    }
    
    inline void check_ret(cublasStatus_t check, std::string op)
    {
      if (check != CUBLAS_STATUS_SUCCESS)
      {
        std::string msg = "cuBLAS " + op + "() failed with error: " + get_cublas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  arma::fmat xtx_gemm(const float alpha, const arma::fmat &x, const int num_gpus)
  {
    const int m = x.n_rows;
    const int n = x.n_cols;
    arma::fmat ret(n, n);
    
    
    cublasStatus_t st;
    
    cublasXtHandle_t h;
    st = cublasXtCreate(&h);
    err::check_ret(st, "xtgemm");
    
    st = cublasXtSetBlockDim(h, blockdim);
    err::check_ret(st, "xtgemm");
    
    int dev_id[num_gpus];
    for (int i=0; i<num_gpus; i++)
      dev_id[i] = i;
    st = cublasXtDeviceSelect(h, num_gpus, dev_id);
    err::check_ret(st, "xtgemm");
    
    st = cublasXtSetCpuRatio(h, CUBLASXT_GEMM, CUBLASXT_FLOAT, cpu_ratio);
    err::check_ret(st, "xtgemm");
    
    const float beta = 0.0f;
    st = cublasXtSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, 
      x.memptr(), m, x.memptr(), m, &beta, ret.memptr(), n);
    err::check_ret(st, "xtgemm");
    
    st = cublasXtDestroy(h);
    err::check_ret(st, "xtgemm");
    
    return ret;
  }
  
  
  
  arma::fmat xtx_syrk(const float alpha, const arma::fmat &x, const int num_gpus)
  {
    const int m = x.n_rows;
    const int n = x.n_cols;
    arma::fmat ret(n, n);
    
    
    cublasStatus_t st;
    
    cublasXtHandle_t h;
    st = cublasXtCreate(&h);
    err::check_ret(st, "xtsyrk");
    
    st = cublasXtSetBlockDim(h, blockdim);
    err::check_ret(st, "xtsyrk");
    
    int dev_id[num_gpus];
    for (int i=0; i<num_gpus; i++)
      dev_id[i] = i;
    st = cublasXtDeviceSelect(h, num_gpus, dev_id);
    err::check_ret(st, "xtsyrk");
    
    st = cublasXtSetCpuRatio(h, CUBLASXT_GEMM, CUBLASXT_FLOAT, cpu_ratio);
    err::check_ret(st, "xtsyrk");
    
    const float beta = 0.0f;
    st = cublasXtSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, m, &alpha, 
      x.memptr(), m, &beta, ret.memptr(), n);
    err::check_ret(st, "xtsyrk");
    
    st = cublasXtDestroy(h);
    err::check_ret(st, "xtsyrk");
    
    cpublas::symmetrize(ret);
    
    return ret;
  }
}
