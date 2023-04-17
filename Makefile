BUILD_DIR ?= ./build
CXXFLAGS ?=
LDFLAGS ?=
RPATHS ?=
LIBXSMM_DIR ?= libxsmm
OPTIONS = -O2 -std=c++20 -pedantic -Wall -Wextra -DTORCH_API_INCLUDE_EXTENSION_H -I.
JSONC_INC = -Isubmodules/json/single_include/
CATCH_INC = -Isubmodules/Catch/single_include/

PYTORCH_INCLUDE ?= $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LINK ?= $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

CXXFLAGS += $(foreach inc,$(PYTORCH_INCLUDE),-isystem$(inc)) -fopenmp
LDFLAGS += $(foreach lin,$(PYTORCH_LINK),-L$(lin)) -ltorch -ltorch_cpu -lc10 ${LIBXSMM_DIR}/lib/libxsmm.a -ldl
RPATHS += $(foreach lin,$(PYTORCH_LINK),-Wl,-rpath,$(lin))

$(info $$CXXFLAGS is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])

${BUILD_DIR}/tpp_nets.a: src/backend/BinaryContraction.cpp src/bench/TensorDot.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -I${LIBXSMM_DIR}/include -c src/backend/BinaryContraction.cpp -o ${BUILD_DIR}/backend/BinaryContraction.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -I${LIBXSMM_DIR}/include ${JSONC_INC} -c src/bench/TensorDot.cpp -o ${BUILD_DIR}/bench/TensorDot.o
		${AR} rcs ${BUILD_DIR}/tpp_nets.a ${BUILD_DIR}/backend/*.o ${BUILD_DIR}/bench/*.o

${BUILD_DIR}/test: ${BUILD_DIR}/tpp_nets.a src/backend/BinaryContraction.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} ${CATCH_INC} -c src/backend/BinaryContraction.test.cpp -o ${BUILD_DIR}/tests/backend/BinaryContraction.test.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} ${CATCH_INC} src/test.cpp ${BUILD_DIR}/tests/backend/*.o ${BUILD_DIR}/tpp_nets.a -o ${BUILD_DIR}/test ${RPATHS} ${LDFLAGS}

${BUILD_DIR}/bench_tdot: ${BUILD_DIR}/tpp_nets.a src/bench_tdot.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} src/bench_tdot.cpp ${BUILD_DIR}/tpp_nets.a -o ${BUILD_DIR}/bench_tdot ${RPATHS} ${LDFLAGS}

test: ${BUILD_DIR}/test
bench: ${BUILD_DIR}/bench_tdot

all: test bench

$(shell mkdir -p ${BUILD_DIR}/bench)
$(shell mkdir -p ${BUILD_DIR}/backend)
$(shell mkdir -p ${BUILD_DIR}/tests/backend)