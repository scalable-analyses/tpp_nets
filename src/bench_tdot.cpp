#include <cstdlib>
#include <iostream>
#include <fstream>
#include "bench/TensorDot.h"

int main( int    i_argc,
          char * i_argv[] ) {
  std::cout << "************************************************" << std::endl;
  std::cout << "*** Running tensordot benchmarking interface ***" << std::endl;
  std::cout << "************************************************" << std::endl;


  if( i_argc != 2 ) {
    std::cerr << "Error, usage: ./bech_tdot my_config.json" << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream l_file( i_argv[1] );
  if( !l_file.good() ) {
    std::cerr << "Error, file does not exist: " << i_argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  l_file.close();

  // vectors holding the configs
  std::vector< std::vector< int64_t > > l_sizes_s;
  std::vector< std::vector< int64_t > > l_sizes_t;
  std::vector< std::vector< int64_t > > l_sizes_u;

  std::vector< std::vector<  int8_t > > l_types_s;
  std::vector< std::vector<  int8_t > > l_types_t;
  std::vector< std::vector<  int8_t > > l_types_u;

  // parse config
  tpp_nets::bench::TensorDot::parse_config( i_argv[1],
                                            l_sizes_s,
                                            l_sizes_t,
                                            l_sizes_u,
                                            l_types_s,
                                            l_types_t,
                                            l_types_u );

  // run settings
  uint64_t l_n_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  for( std::size_t l_co = 0; l_co < l_sizes_s.size(); l_co++ ) {
    std::cout << "*** setting " << l_co+1 << " of " << l_sizes_s.size() << " ***" << std::endl;
    std::cout << "config:" << std::endl;
    std::cout << "  sizes_s:";
    for( std::size_t l_en = 0; l_en < l_sizes_s[l_co].size(); l_en++ ) {
      std::cout << " " << l_sizes_s[l_co][l_en];
    }
    std::cout << std::endl;

    std::cout << "  sizes_t:";
    for( std::size_t l_en = 0; l_en < l_sizes_t[l_co].size(); l_en++ ) {
      std::cout << " " << l_sizes_t[l_co][l_en];
    }
    std::cout << std::endl;

    std::cout << "  sizes_u:";
    for( std::size_t l_en = 0; l_en < l_sizes_u[l_co].size(); l_en++ ) {
      std::cout << " " << l_sizes_u[l_co][l_en];
    }
    std::cout << std::endl;

    std::cout << "  types_s:";
    for( std::size_t l_en = 0; l_en < l_sizes_s[l_co].size(); l_en++ ) {
      std::cout << " " << (int) l_types_s[l_co][l_en];
    }
    std::cout << std::endl;

    std::cout << "  types_t:";
    for( std::size_t l_en = 0; l_en < l_sizes_t[l_co].size(); l_en++ ) {
      std::cout << " " << (int) l_types_t[l_co][l_en];
    }
    std::cout << std::endl;

    std::cout << "  types_u:";
    for( std::size_t l_en = 0; l_en < l_sizes_u[l_co].size(); l_en++ ) {
      std::cout << " " << (int) l_types_u[l_co][l_en];
    }
    std::cout << std::endl;

    for( int8_t l_kernel_type = 0; l_kernel_type < 2; l_kernel_type++ ) {
      if( l_kernel_type == 0 ) {
        std::cout << "tppdot:" << std::endl;

        bool l_correct = tpp_nets::bench::TensorDot::check( l_sizes_s[l_co],
                                                            l_sizes_t[l_co],
                                                            l_sizes_u[l_co],
                                                            l_types_s[l_co],
                                                            l_types_t[l_co],
                                                            l_types_u[l_co] );
        std::cout << "  correctness: " << l_correct << std::endl;
      }
      else if( l_kernel_type == 1) {
        std::cout << "at::tensordot:" << std::endl;
      }
 
      std::tie( l_n_repetitions,
                l_time,
                l_gflops ) = tpp_nets::bench::TensorDot::perf( l_kernel_type,
                                                               l_sizes_s[l_co],
                                                               l_sizes_t[l_co],
                                                               l_sizes_u[l_co],
                                                               l_types_s[l_co],
                                                               l_types_t[l_co],
                                                               l_types_u[l_co] );

      std::cout << "  repetitions: " << l_n_repetitions << std::endl;
      std::cout << "  duration: " << l_time << " seconds" << std::endl;
      std::cout << "  GFLOPS: " << l_gflops << std::endl;
    }

    std::cout << std::endl;
  }

  std::cout << "****************" << std::endl;
  std::cout << "*** finished ***" << std::endl;
  std::cout << "****************" << std::endl;

  return EXIT_SUCCESS;
}