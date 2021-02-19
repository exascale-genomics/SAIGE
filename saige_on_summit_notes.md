# SAIGE on Summit

## Pre-reqs

```bash
# NOTE: set path as appropriate
export ROOT="/gpfs/alpine/stf011/proj-shared/va8/saige/"
mkdir -p ${ROOT}
cd ${ROOT}
export RLIB="${ROOT}/lib"
mkdir -p ${RLIB}


export MAKE="/usr/bin/make -j 32"

module load gcc/6.4.0
module load curl
module load bzip2
module load openblas

module load python
module load cmake
module load mercurial
```


## Building R

```bash
export RVER="4.0.3"

wget https://cran.r-project.org/src/base/R-4/R-${RVER}.tar.gz
tar -zxf R-${RVER}.tar.gz 
cd R-${RVER}

export OBPATH=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/openblas-0.3.9-g7tkwn4kc7ukzxqqqia4jwpqr27aw3cu/lib
./configure --with-blas=${OBPATH}/libopenblas.so --with-lapack=${OBPATH}/libopenblas.so \
  --with-x=no \
  --enable-R-shlib=yes \
  --enable-memory-profiling=no \
  --with-pcre1

make -j

export REXEC="${ROOT}/R-${RVER}/bin/R"
export RSCRIPT="${ROOT}/R-${RVER}/bin/Rscript"
```


## Building TBB and RcppParallel

```bash
cd ${ROOT}

wget https://github.com/oneapi-src/oneTBB/archive/4.4.5.tar.gz
tar zxf 4.4.5.tar.gz
cd oneTBB-4.4.5/
make -j

# next 3 must be set for RcppParallel
export TBB="${ROOT}/oneTBB-4.4.5/"
export TBB_INC="${TBB}/include"
export TBB_LIB="${TBB}/build/linux_ppc64le_gcc_cc6.4.0_libc2.17_kernel4.14.0_release/"

mv ${TBB_LIB}/libtbbmalloc_proxy.so.2 ${TBB_LIB}/libtbbmalloc_proxy.so
mv ${TBB_LIB}/libtbbmalloc.so.2 ${TBB_LIB}/libtbbmalloc.so
mv ${TBB_LIB}/libtbb.so.2 ${TBB_LIB}/libtbb.so

wget https://github.com/RcppCore/RcppParallel/archive/master.zip
unzip master.zip
${REXEC} CMD INSTALL RcppParallel-master --library=${RLIB}
```


## Building SAIGE

R pre-reqs

```bash
cd ${ROOT}

${REXEC}script -e "install.packages(c(\
  'R.utils', 'Rcpp', 'data.table', 'RcppEigen', 'Matrix', 'BH', \
  'optparse', 'SPAtest', 'SKAT'), lib=Sys.getenv('RLIB'), repos='https://cran.rstudio.com')"

wget https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.9.900.1.0.tar.gz
${REXEC} CMD INSTALL --library=${RLIB} RcppArmadillo_0.9.900.1.0.tar.gz

wget https://cran.r-project.org/src/contrib/Archive/MetaSKAT/MetaSKAT_0.80.tar.gz
${REXEC} CMD INSTALL --library=${RLIB} MetaSKAT_0.80.tar.gz
```

download SAIGE

```bash
export LC_ALL=en_US.UTF-8

git clone --depth 1 -b master https://github.com/exascale-genomics/SAIGE
rm -rf ./SAIGE/configure
rm -rf ./SAIGE/src/*.o ./SAIGE/src/*.so
rm -rf ./SAIGE/thirdParty/cget

pip install --user cget
export PATH=${PATH}:${HOME}/.local/bin/
```

bgen and its reqs - make the changes listed in the NOTE

```bash
mkdir -p ./SAIGE/thirdParty/cget
CXX=g++ CC=gcc cget install -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" --prefix ./SAIGE/thirdParty/cget xiaoyeli/superlu
CXX=g++ CC=gcc cget install -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" --prefix ./SAIGE/thirdParty/cget https://github.com/statgen/savvy/archive/v1.3.0.tar.gz

cd ./SAIGE/thirdParty/bgen
sed -i -e 1's/$/2 &/' waf
./waf configure
./waf
cd ../../..
```

build SAIGE - make the changes listed in the NOTE

```
${REXEC} CMD INSTALL SAIGE --library=${RLIB}
```


# Tests

## OpenBLAS + OpenMP test

test.r

```r
x = matrix(rnorm(1e7),  nrow = 1e4)
system.time(crossprod(x))
```

Running the test

```bash
for nth in 1 2 4 8 16; do
  jsrun -n1 -c42 -bpacked:42 -EOMP_NUM_THREADS=${nth} ${RSCRIPT} test.r
done
```

Expected outcome if working correctly: different runtimes


## RcppParallel Basic Test

test.r

```r
library("RcppParallel", lib.loc=Sys.getenv('RLIB'))
defaultNumThreads()
```

Running the test

```bash
jsrun -n1 -c42 -bpacked:42 -EOMP_NUM_THREADS=42 ${RSCRIPT} test.r
```

Expected outcome if working correctly: a number other than 4


## RcppParallel Runtime Test

test.r

```r
.libPaths(Sys.getenv('RLIB'))
library(Rcpp)

src = "
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
using namespace RcppParallel;

struct SquareRoot : public Worker
{
   // source matrix
   const RMatrix<double> input;
   
   // destination matrix
   RMatrix<double> output;
   
   // initialize with source and destination
   SquareRoot(const NumericMatrix input, NumericMatrix output) 
      : input(input), output(output) {}
   
   // take the square root of the range of elements requested
   void operator()(std::size_t begin, std::size_t end) {
      std::transform(input.begin() + begin, 
                     input.begin() + end, 
                     output.begin() + begin, 
                     ::sqrt);
   }
};

// [[Rcpp::export]]
NumericMatrix parallelMatrixSqrt(NumericMatrix x) {
  
  // allocate the output matrix
  NumericMatrix output(x.nrow(), x.ncol());
  
  // SquareRoot functor (pass input and output matrixes)
  SquareRoot squareRoot(x, output);
  
  // call parallelFor to do the work
  parallelFor(0, x.length(), squareRoot);
  
  // return the output matrix
  return output;
}
"

sourceCpp(code=src)
RcppParallel::defaultNumThreads()

n = 7500
x = matrix(runif(n*n), n, n)

f = function(x, n) {
  RcppParallel::setThreadOptions(n)
  system.time(parallelMatrixSqrt(x))
}

f(x, 1)
f(x, 2)
f(x, 4)
f(x, 6)
f(x, 8)
f(x, 16)
```

Running the test

```bash
jsrun -n1 -c42 -bpacked:42 -EOMP_NUM_THREADS=42 ${RSCRIPT} test.r
```

Expected outcome if working correctly: different runtimes
