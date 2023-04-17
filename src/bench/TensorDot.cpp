#include <fstream>
#include "TensorDot.h"
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include "../backend/BinaryContraction.h"

bool tpp_nets::bench::TensorDot::check( std::vector< int64_t > i_sizes_s,
                                        std::vector< int64_t > i_sizes_t,
                                        std::vector< int64_t > i_sizes_u,
                                        std::vector<  int8_t > i_types_s,
                                        std::vector<  int8_t > i_types_t,
                                        std::vector<  int8_t > i_types_u ) {
  // compute solution through tppdot
  int64_t l_n_dims_s = i_sizes_s.size();
  int64_t l_n_dims_t = i_sizes_t.size();
  int64_t l_n_dims_u = i_sizes_u.size();

  at::Tensor l_s = at::rand(  i_sizes_s );
  at::Tensor l_t = at::rand(  i_sizes_t );
  at::Tensor l_u = at::zeros( i_sizes_u );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( l_n_dims_s,
                     l_n_dims_t,
                     l_n_dims_u,
                     i_sizes_s.data(),
                     i_sizes_t.data(),
                     i_types_s.data(),
                     i_types_t.data(),
                     i_types_u.data(),
                     l_strides_s.data(),
                     l_strides_t.data(),
                     l_strides_u.data(),
                     l_s.data_ptr(),
                     l_t.data_ptr(),
                     l_u.data_ptr() );

  // compute solution through ATen's tensordot
  std::vector< int64_t > l_dims_reduction_s;
  std::vector< int64_t > l_dims_reduction_t;

  for( std::size_t l_di_s = 0; l_di_s < i_types_s.size(); l_di_s++ ) {
    if( i_types_s[l_di_s] == 1 ) {
      l_dims_reduction_s.push_back( l_di_s );
    }
  }

  for( std::size_t l_di_t = 0; l_di_t < i_types_s.size(); l_di_t++ ) {
    if( i_types_t[l_di_t] == 1 ) {
      l_dims_reduction_t.push_back( l_di_t );
    }
  }

  at::Tensor l_ref = at::tensordot( l_s,
                                    l_t,
                                    l_dims_reduction_s,
                                    l_dims_reduction_t );

  // derive required permutation
  std::vector< int64_t > l_perm;
  for( std::size_t l_di_u = 0; l_di_u < i_types_u.size(); l_di_u++ ) {
    if( i_types_u[l_di_u] == 0 ) {
      l_perm.push_back( l_di_u );
    }
  }
  for( std::size_t l_di_u = 0; l_di_u < i_types_u.size(); l_di_u++ ) {
    if( i_types_u[l_di_u] == 1 ) {
      l_perm.push_back( l_di_u );
    }
  }

  l_u = l_u.permute( l_perm );

  return at::allclose( l_u, l_ref );
}

double tpp_nets::bench::TensorDot::time_aten( std::vector< int64_t > i_sizes_s,
                                              std::vector< int64_t > i_sizes_t,
                                              std::vector<  int8_t > i_types_s,
                                              std::vector<  int8_t > i_types_t,
                                              int64_t                i_n_repetitions ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  at::Tensor l_s = at::rand(  i_sizes_s );
  at::Tensor l_t = at::rand(  i_sizes_t );

  std::vector< int64_t > l_dims_reduction_s;
  std::vector< int64_t > l_dims_reduction_t;

  for( std::size_t l_di_s = 0; l_di_s < i_types_s.size(); l_di_s++ ) {
    if( i_types_s[l_di_s] == 1 ) {
      l_dims_reduction_s.push_back( l_di_s );
    }
  }

  for( std::size_t l_di_t = 0; l_di_t < i_types_s.size(); l_di_t++ ) {
    if( i_types_t[l_di_t] == 1 ) {
      l_dims_reduction_t.push_back( l_di_t );
    }
  }

  // warmup
  at::tensordot( l_s,
                 l_t,
                 l_dims_reduction_s,
                 l_dims_reduction_t );

  // benchmark
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( int64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    at::tensordot( l_s,
                   l_t,
                   l_dims_reduction_s,
                   l_dims_reduction_t );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}

double tpp_nets::bench::TensorDot::time_tppdot( std::vector< int64_t > i_sizes_s,
                                                std::vector< int64_t > i_sizes_t,
                                                std::vector< int64_t > i_sizes_u,
                                                std::vector<  int8_t > i_types_s,
                                                std::vector<  int8_t > i_types_t,
                                                std::vector<  int8_t > i_types_u,
                                                int64_t                i_n_repetitions ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  int64_t l_n_dims_s = i_sizes_s.size();
  int64_t l_n_dims_t = i_sizes_t.size();
  int64_t l_n_dims_u = i_sizes_u.size();

  at::Tensor l_s = at::rand(  i_sizes_s );
  at::Tensor l_t = at::rand(  i_sizes_t );
  at::Tensor l_u = at::zeros( i_sizes_u );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  // warmup
  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( l_n_dims_s,
                    l_n_dims_t,
                    l_n_dims_u,
                    i_sizes_s.data(),
                    i_sizes_t.data(),
                    i_types_s.data(),
                    i_types_t.data(),
                    i_types_u.data(),
                    l_strides_s.data(),
                    l_strides_t.data(),
                    l_strides_u.data(),
                    l_s.data_ptr(),
                    l_t.data_ptr(),
                    l_u.data_ptr() );

  // benchmark
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( int64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    l_bin_con.tppdot( l_n_dims_s,
                      l_n_dims_t,
                      l_n_dims_u,
                      i_sizes_s.data(),
                      i_sizes_t.data(),
                      i_types_s.data(),
                      i_types_t.data(),
                      i_types_u.data(),
                      l_strides_s.data(),
                      l_strides_t.data(),
                      l_strides_u.data(),
                      l_s.data_ptr(),
                      l_t.data_ptr(),
                      l_u.data_ptr() );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}


std::tuple< uint64_t,
            double,
            double > tpp_nets::bench::TensorDot::perf( int8_t                 i_kernel_type,
                                                       std::vector< int64_t > i_sizes_s,
                                                       std::vector< int64_t > i_sizes_t,
                                                       std::vector< int64_t > i_sizes_u,
                                                       std::vector<  int8_t > i_types_s,
                                                       std::vector<  int8_t > i_types_t,
                                                       std::vector<  int8_t > i_types_u,
                                                       double                 i_time_target,
                                                       uint64_t               i_n_repetitions_initial ) {
  // get number of threads and print
  int l_n_threads = 1;

  // get number of flops per iter
  int64_t l_n_flops = 2;
  for( std::size_t l_di_s = 0; l_di_s < i_sizes_s.size(); l_di_s++ ) {
    l_n_flops *= i_sizes_s[l_di_s]; // M and K
  }
  for( std::size_t l_di_t = 0; l_di_t < i_sizes_t.size(); l_di_t++ ) {
    if( i_types_t[l_di_t] == 0 ) {
      l_n_flops *= i_sizes_t[l_di_t]; // N
    }
  }

  double l_dur = 0;
  if( i_kernel_type == 0 ) {
    // get time required for initial number of reps
    l_dur = time_tppdot( i_sizes_s,
                         i_sizes_t,
                         i_sizes_u,
                         i_types_s,
                         i_types_t,
                         i_types_u,
                         i_n_repetitions_initial );
  }
  else if( i_kernel_type == 1 ) {
    l_dur = time_aten( i_sizes_s,
                       i_sizes_t,
                       i_types_s,
                       i_types_t,
                       i_n_repetitions_initial );
  }
  else {
    assert( false );
  }

  // derive number of reps for targeted duration
  double l_scaling_time = i_time_target / l_dur;
  uint64_t l_n_repetitions_adj = i_n_repetitions_initial * l_scaling_time;
  if( l_n_repetitions_adj == 0 ) {
    l_n_repetitions_adj = 1;
  }

  // benchmark kernel
  if( i_kernel_type == 0 ) {
    l_dur = time_tppdot( i_sizes_s,
                         i_sizes_t,
                         i_sizes_u,
                         i_types_s,
                         i_types_t,
                         i_types_u,
                         l_n_repetitions_adj );
  }
  else if( i_kernel_type == 1 ) {
    l_dur = time_aten( i_sizes_s,
                       i_sizes_t,
                       i_types_s,
                       i_types_t,
                       l_n_repetitions_adj );
  }
  else {
    assert( false );
  }

  // derive gflops
  double l_gflops = l_n_repetitions_adj;
  l_gflops *= l_n_flops / l_dur;
  l_gflops *= l_n_threads;
  l_gflops *= 1.0E-9;

  return std::make_tuple( l_n_repetitions_adj,
                          l_dur,
                          l_gflops );
}

void tpp_nets::bench::TensorDot:: parse_config( std::string                             i_path,
                                                std::vector< std::vector< int64_t > > & o_sizes_s,
                                                std::vector< std::vector< int64_t > > & o_sizes_t,
                                                std::vector< std::vector< int64_t > > & o_sizes_u,
                                                std::vector< std::vector<  int8_t > > & o_types_s,
                                                std::vector< std::vector<  int8_t > > & o_types_t,
                                                std::vector< std::vector<  int8_t > > & o_types_u ) {
  // reset configs
  o_sizes_s.resize(0);
  o_sizes_t.resize(0);
  o_sizes_u.resize(0);

  o_types_s.resize(0);
  o_types_t.resize(0);
  o_types_u.resize(0);

  // parse json file
  std::ifstream l_file( i_path );
  nlohmann::json l_data = nlohmann::json::parse( l_file );

  // store configs
  for( std::size_t l_co = 0; l_co < l_data.size(); l_co++ ) {
    o_sizes_s.push_back(l_data[l_co]["sizes_s"] );
    o_sizes_t.push_back(l_data[l_co]["sizes_t"] );
    o_sizes_u.push_back(l_data[l_co]["sizes_u"] );

    o_types_s.push_back(l_data[l_co]["types_s"] );
    o_types_t.push_back(l_data[l_co]["types_t"] );
    o_types_u.push_back(l_data[l_co]["types_u"] );
  }
}