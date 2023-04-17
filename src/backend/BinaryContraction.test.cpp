#include <catch2/catch.hpp>
#include <ATen/ATen.h>
#include "BinaryContraction.h"


#include <iostream>

TEST_CASE( "Tests the tppdot routine with column-major A, row-major B, and column-major C.",
           "[tpp_nets][BinaryContraction][tppdot_000]" ) {
  //                        0   1   2   3
  //                       k0  m0  k1  m1
  //                        a   b   c   d
  int64_t l_sizes_s[4] = { 17,  5, 22, 13 };

  //                        0    1   2   3
  //                       k0   n0  k1  n1
  //                        a    e   c   f
  int64_t l_sizes_t[4] = { 17,   8, 22, 7 };

  at::Tensor l_s = at::rand( l_sizes_s );
  at::Tensor l_t = at::rand( l_sizes_t );
  //                            0   1   2   3
  //                           n0  m0  n1  m1
  //                            e   b   f   d
  at::Tensor l_u = at::zeros( { 8,  5,  7, 13 } );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  // 0: m/n
  // 1: k
  // 2: b
  int8_t l_types_s[4] = { 1, 0, 1, 0 };
  int8_t l_types_t[4] = { 1, 0, 1, 0 };
  // 0: m
  // 1: n
  int8_t l_types_u[4] = { 1, 0, 1, 0 };

  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( 4,
                    4,
                    4,
                    l_sizes_s,
                    l_sizes_t,
                    l_types_s,
                    l_types_t,
                    l_types_u,
                    l_strides_s.data(),
                    l_strides_t.data(),
                    l_strides_u.data(),
                    l_s.data_ptr(),
                    l_t.data_ptr(),
                    l_u.data_ptr() );

  // einsum reference
  at::Tensor l_ref = at::einsum( "abcd,aecf->ebfd",
                                 {l_s, l_t} );

  REQUIRE( at::allclose( l_u,
                         l_ref ) );

  // tensordot reference
  at::Tensor l_ref_td = at::tensordot( l_s,
                                       l_t,
                                       { 0, 2 },
                                       { 0, 2 } );

  l_ref_td = l_ref_td.permute( {2, 0, 3, 1} );

  REQUIRE( at::allclose( l_u,
                         l_ref_td ) );

}

TEST_CASE( "Tests the tppdot routine with column-major A, B and C.",
           "[tpp_nets][BinaryContraction][tppdot_010]" ) {
  //                        0   1   2   3
  //                       k0  m0  k1  m1
  //                        a   b   c   d
  int64_t l_sizes_s[4] = { 17,  5, 22, 13 };

  //                        0    1   2   3
  //                       n0   k0  n1  k1
  //                        e    a   f   c
  int64_t l_sizes_t[4] = {  8,  17,  7, 22 };

  at::Tensor l_s = at::rand( l_sizes_s );
  at::Tensor l_t = at::rand( l_sizes_t );
  //                            0   1   2   3
  //                           n0  m0  n1  m1
  //                            e   b   f   d
  at::Tensor l_u = at::zeros( { 8,  5,  7, 13 } );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  // 0: m/n
  // 1: k
  // 2: b
  int8_t l_types_s[4] = { 1, 0, 1, 0 };
  int8_t l_types_t[4] = { 0, 1, 0, 1 };
  // 0: m
  // 1: n
  int8_t l_types_u[4] = { 1, 0, 1, 0 };

  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( 4,
                    4,
                    4,
                    l_sizes_s,
                    l_sizes_t,
                    l_types_s,
                    l_types_t,
                    l_types_u,
                    l_strides_s.data(),
                    l_strides_t.data(),
                    l_strides_u.data(),
                    l_s.data_ptr(),
                    l_t.data_ptr(),
                    l_u.data_ptr() );

  // einsum reference
  at::Tensor l_ref = at::einsum( "abcd,eafc->ebfd",
                                 {l_s, l_t} );

  REQUIRE( at::allclose( l_u,
                         l_ref ) );

  // tensordot reference
  at::Tensor l_ref_td = at::tensordot( l_s,
                                       l_t,
                                       { 0, 2 },
                                       { 1, 3 } );

  l_ref_td = l_ref_td.permute( {2, 0, 3, 1} );

  REQUIRE( at::allclose( l_u,
                         l_ref_td ) );

}

TEST_CASE( "Tests the tppdot routine with row-major A, and column-major B and C.",
           "[tpp_nets][BinaryContraction][tppdot_110]" ) {
  //                        0   1   2   3
  //                       m0  k0  m1  k1
  //                        a   b   c   d
  int64_t l_sizes_s[4] = {  5, 17, 13, 22 };

  //                        0    1   2   3
  //                       n0   k0  n1  k1
  //                        e    b   f   d
  int64_t l_sizes_t[4] = {  8,  17,  7, 22 };

  at::Tensor l_s = at::rand( l_sizes_s );
  at::Tensor l_t = at::rand( l_sizes_t );
  //                            0   1   2   3
  //                           n0  m0  n1  m1
  //                            e   a   f   c
  at::Tensor l_u = at::zeros( { 8,  5,  7, 13 } );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  // 0: m/n
  // 1: k
  // 2: b
  int8_t l_types_s[4] = { 0, 1, 0, 1 };
  int8_t l_types_t[4] = { 0, 1, 0, 1 };
  // 0: m
  // 1: n
  int8_t l_types_u[4] = { 1, 0, 1, 0 };

  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( 4,
                    4,
                    4,
                    l_sizes_s,
                    l_sizes_t,
                    l_types_s,
                    l_types_t,
                    l_types_u,
                    l_strides_s.data(),
                    l_strides_t.data(),
                    l_strides_u.data(),
                    l_s.data_ptr(),
                    l_t.data_ptr(),
                    l_u.data_ptr() );

  // einsum reference
  at::Tensor l_ref = at::einsum( "abcd,ebfd->eafc",
                                 {l_s, l_t} );

  REQUIRE( at::allclose( l_u,
                         l_ref ) );

  // tensordor reference
  at::Tensor l_ref_td = at::tensordot( l_s,
                                       l_t,
                                       { 1, 3 },
                                       { 1, 3 } );

  l_ref_td = l_ref_td.permute( {2, 0, 3, 1} );

  REQUIRE( at::allclose( l_u,
                         l_ref_td ) );
}

TEST_CASE( "Tests the tppdot routine with row-major A and B, and column-major C.",
           "[tpp_nets][BinaryContraction][tppdot_100]" ) {
  //                        0   1   2   3
  //                       m0  k0  m1  k1
  //                        a   b   c   d
  int64_t l_sizes_s[4] = {  5, 17, 13, 22 };

  //                        0    1   2   3
  //                       k0   n0  k1  n1
  //                        b    e   d   f
  int64_t l_sizes_t[4] = { 17,   8, 22,  7 };

  at::Tensor l_s = at::rand( l_sizes_s );
  at::Tensor l_t = at::rand( l_sizes_t );
  //                            0   1   2   3
  //                           n0  m0  n1  m1
  //                            e   a   f   c
  at::Tensor l_u = at::zeros( { 8,  5,  7, 13 } );

  std::vector< int64_t > l_strides_s = l_s.strides().vec();
  std::vector< int64_t > l_strides_t = l_t.strides().vec();
  std::vector< int64_t > l_strides_u = l_u.strides().vec();

  // 0: m/n
  // 1: k
  // 2: b
  int8_t l_types_s[4] = { 0, 1, 0, 1 };
  int8_t l_types_t[4] = { 1, 0, 1, 0 };
  // 0: m
  // 1: n
  int8_t l_types_u[4] = { 1, 0, 1, 0 };

  tpp_nets::backend::BinaryContraction l_bin_con;
  l_bin_con.tppdot( 4,
                    4,
                    4,
                    l_sizes_s,
                    l_sizes_t,
                    l_types_s,
                    l_types_t,
                    l_types_u,
                    l_strides_s.data(),
                    l_strides_t.data(),
                    l_strides_u.data(),
                    l_s.data_ptr(),
                    l_t.data_ptr(),
                    l_u.data_ptr() );

  // einsum reference
  at::Tensor l_ref = at::einsum( "abcd,bedf->eafc",
                                 {l_s, l_t} );

  REQUIRE( at::allclose( l_u,
                         l_ref ) );

  // tensordor reference
  at::Tensor l_ref_td = at::tensordot( l_s,
                                       l_t,
                                       { 1, 3 },
                                       { 0, 2 } );

  l_ref_td = l_ref_td.permute( {2, 0, 3, 1} );

  REQUIRE( at::allclose( l_u,
                         l_ref_td ) );
}