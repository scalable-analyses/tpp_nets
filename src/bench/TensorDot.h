#ifndef TPP_NETS_BENCH_TENSOR_DOT

#include <cstdint>
#include <vector>
#include <tuple>
#include <string>

namespace tpp_nets {
  namespace bench {
    class TensorDot;
  }
}

class tpp_nets::bench::TensorDot {
  private:
    /**
     * Measures the performance (time) of ATen's tensordot(S, T):
     *
     * The routine is executed repeatedly as specified by the input i_n_repetitions.
     * 
     * @param i_sizes_s sizes of S's dimensions.
     * @param i_sizes_t sizes of T's dimensions.
     * @param i_types_s types of S's dimensions.
     * @param i_types_t types of T's dimensions. 
     * @param i_n_repetitions number of performed repetitions.
     * @return duration in seconds.
     **/
    static double time_aten( std::vector< int64_t > i_sizes_s,
                             std::vector< int64_t > i_sizes_t,
                             std::vector<  int8_t > i_types_s,
                             std::vector<  int8_t > i_types_t,
                             int64_t                i_n_repetitions );


    /**
     * Measures the performance (time) of tppdot:
     * U += contract(S, T).
     *
     * The routine is executed repeatedly as specified by the input i_n_repetitions.
     *
     * @param i_sizes_s sizes of S's dimensions.
     * @param i_sizes_t sizes of T's dimensions.
     * @param i_sizes_u sizes of U's dimension.
     * @param i_types_s types of S's dimensions.
     * @param i_types_t types of T's dimensions. 
     * @param i_n_repetitions number of performed repetitions.
     * @return duration in seconds.
     **/
    static double time_tppdot( std::vector< int64_t > i_sizes_s,
                               std::vector< int64_t > i_sizes_t,
                               std::vector< int64_t > i_sizes_u,
                               std::vector<  int8_t > i_types_s,
                               std::vector<  int8_t > i_types_t,
                               std::vector<  int8_t > i_types_u,
                               int64_t                i_n_repetitions );

  public:
    /**
     * Parses a JSON config using the given path.
     *
     * @param i_path path of the JSON config from which the settings are read.
     * @param o_sizes_s will be set to dimension sizes of S.
     * @param o_sizes_t will be set to dimension sizes of T.
     * @param o_sizes_u will be set to dimension sizes of U.
     * @param o_types_s will be set to dimension types of S.
     * @param o_types_t will be set to dimension types of T.
     * @param o_types_u will be set to dimension types of U.
     **/
    static void parse_config( std::string                             i_path,
                              std::vector< std::vector< int64_t > > & o_sizes_s,
                              std::vector< std::vector< int64_t > > & o_sizes_t,
                              std::vector< std::vector< int64_t > > & o_sizes_u,
                              std::vector< std::vector<  int8_t > > & o_types_s,
                              std::vector< std::vector<  int8_t > > & o_types_t,
                              std::vector< std::vector<  int8_t > > & o_types_u );

    /**
     * Check the correctness of the tppdot routine by comparing it to aten::tensordot.
     *
     * @param i_sizes_s will be set to dimension sizes of S.
     * @param i_sizes_t will be set to dimension sizes of T.
     * @param i_sizes_u will be set to dimension sizes of U.
     * @param i_types_s will be set to dimension types of S.
     * @param i_types_t will be set to dimension types of T.
     * @param i_types_u will be set to dimension types of U.
     * @return true if the same (up to an epsilon, using at::allclose) tensors are computed, false otherwise.
     **/
    static bool check( std::vector< int64_t > i_sizes_s,
                       std::vector< int64_t > i_sizes_t,
                       std::vector< int64_t > i_sizes_u,
                       std::vector<  int8_t > i_types_s,
                       std::vector<  int8_t > i_types_t,
                       std::vector<  int8_t > i_types_u );

    /**
     * Benchmarks the performance (repetitions, time, gflops) of the given tensordot implementation.
     *
     * @param i_kernel_type benchmarked kernel, 0: tppdot, 1: at::tensordor.
     * @param i_sizes_s will be set to dimension sizes of S.
     * @param i_sizes_t will be set to dimension sizes of T.
     * @param i_sizes_u will be set to dimension sizes of U.
     * @param i_types_s will be set to dimension types of S.
     * @param i_types_t will be set to dimension types of T.
     * @param i_types_u will be set to dimension types of U.
     * @param i_time_target targeted total execution time; the number of actual repetitions is adjusted accordingly.
     * @param i_n_repetitions_initial initial number of performed repetitions.
     * @return (repetitions, time, gflops).
     **/
    static std::tuple< uint64_t,
                       double,
                       double > perf( int8_t                 i_kernel_type,
                                      std::vector< int64_t > i_sizes_s,
                                      std::vector< int64_t > i_sizes_t,
                                      std::vector< int64_t > i_sizes_u,
                                      std::vector<  int8_t > i_types_s,
                                      std::vector<  int8_t > i_types_t,
                                      std::vector<  int8_t > i_types_u,
                                      double                 i_time_target = 10.0,
                                      uint64_t               i_n_repetitions_initial = 10 );
};

#endif