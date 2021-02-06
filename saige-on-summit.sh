#!/bin/sh

# change this VVVVVVVVVVVVVVVVVVVVVVVVV
#cd /gpfs/alpine/stf011/proj-shared/va8/
cd /gpfs/alpine/proj-shared/med112/
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

export MAKE="/usr/bin/make -j 16"

module swap xl gcc/4.8.5
module load r/3.5.3
module load python
module load cmake
module load mercurial
module load openblas

mkdir saige
cd saige
mkdir lib

wget https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.9.900.1.0.tar.gz
R CMD INSTALL --library=./lib RcppArmadillo_0.9.900.1.0.tar.gz

Rscript -e "install.packages(c(\
  'R.utils', 'Rcpp', 'RcppParallel', 'data.table', 'RcppEigen', 'Matrix', 'BH', \
  'optparse', 'SPAtest', 'SKAT'), lib='./lib', repos='https://cran.rstudio.com')"

wget https://cran.r-project.org/src/contrib/Archive/MetaSKAT/MetaSKAT_0.80.tar.gz
R CMD INSTALL --library=./lib MetaSKAT_0.80.tar.gz

export LC_ALL=en_US.UTF-8

git clone --depth 1 -b master https://github.com/weizhouUMICH/SAIGE
rm -rf ./SAIGE/configure
rm -rf ./SAIGE/src/*.o ./SAIGE/src/*.so
rm -rf ./SAIGE/thirdParty/cget

pip install --user cget
export PATH=${PATH}:${HOME}/.local/bin/

mkdir -p ./SAIGE/thirdParty/cget
cget install -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" --prefix ./SAIGE/thirdParty/cget statgen/savvy
cget install -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC" --prefix ./SAIGE/thirdParty/cget xiaoyeli/superlu

cd ./SAIGE/thirdParty/bgen
# NOTE: change first line of ./waf to use python2, i.e. make it: '#!/usr/bin/env python2'
./waf configure
./waf
cd ../../..

# NOTE: change ./SAIGE/src/Makevars line 21 to include '-lopenblas' before $(LAPACK_LIBS)
R CMD INSTALL SAIGE --library=./lib
