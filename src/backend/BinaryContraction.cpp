#include <cassert>
#include <libxsmm.h>
#include "BinaryContraction.h"

int64_t tpp_nets::backend::BinaryContraction::filter_attributes( int64_t         i_size,
                                                                 int8_t          i_type_filter,
                                                                 int8_t  const * i_types,
                                                                 int64_t const * i_attributes,
                                                                 int64_t       * o_attributes ) {
  int64_t l_id_out = 0;
  for( int64_t l_id_in = 0; l_id_in < i_size; l_id_in++ ) {
    if( i_types[l_id_in] == i_type_filter ) {
      o_attributes[l_id_out] = i_attributes[l_id_in];
      l_id_out++;
    }
  }
  return l_id_out;
}

int64_t tpp_nets::backend::BinaryContraction::loop_configs( int64_t         i_n_dims_a,
                                                            int64_t         i_n_dims_b,
                                                            int8_t          i_type_filter_a,
                                                            int8_t          i_type_filter_b,
                                                            int8_t  const * i_types_a,
                                                            int8_t  const * i_types_b,
                                                            int64_t const * i_sizes_a,
                                                            int64_t const * i_strides_a,
                                                            int64_t const * i_strides_b,
                                                            int64_t       * o_loops_sizes,
                                                            int64_t       * o_loops_strides_a,
                                                            int64_t       * o_loops_strides_b ) {
  int64_t l_num_loops       = filter_attributes( i_n_dims_a,
                                                  i_type_filter_a,
                                                  i_types_a,
                                                  i_sizes_a,
                                                  o_loops_sizes );

  int64_t l_num_loops_tmp_0 = filter_attributes( i_n_dims_a,
                                                  i_type_filter_a,
                                                  i_types_a,
                                                  i_strides_a,
                                                  o_loops_strides_a );
  assert( l_num_loops == l_num_loops_tmp_0 );

  int64_t l_num_loops_tmp_1 = filter_attributes( i_n_dims_b,
                                                  i_type_filter_b,
                                                  i_types_b,
                                                  i_strides_b,
                                                  o_loops_strides_b );
  assert( l_num_loops == l_num_loops_tmp_1 );

  return l_num_loops;
}

bool tpp_nets::backend::BinaryContraction::advance_loop( int64_t         i_num_loops,
                                                         int64_t const * i_sizes,
                                                         int64_t       * io_counters ) {
  for( int64_t l_id_loop = i_num_loops; l_id_loop >= 0; l_id_loop-- ) {
    if( io_counters[l_id_loop]+1 < i_sizes[l_id_loop] ) {
      io_counters[l_id_loop]++;
      return false;
    }
    else {
      io_counters[l_id_loop] = 0;
    }
  }

  return true;
}

void tpp_nets::backend::BinaryContraction::tppdot( int64_t         i_n_dims_s,
                                                   int64_t         i_n_dims_t,
                                                   int64_t         i_n_dims_u,
                                                   int64_t const * i_sizes_s,
                                                   int64_t const * i_sizes_t,
                                                   int8_t  const * i_types_s,
                                                   int8_t  const * i_types_t,
                                                   int8_t  const * i_types_u,
                                                   int64_t const * i_strides_s,
                                                   int64_t const * i_strides_t,
                                                   int64_t const * i_strides_u,
                                                   void          * i_s,
                                                   void          * i_t,
                                                   void          * o_u ) {
  int64_t l_dtype_size = 4;

  // create LIBXSMM kernel
  assert( i_strides_s[i_n_dims_s-1] == 1 );
  assert( i_strides_t[i_n_dims_t-1] == 1 );
  assert( i_strides_u[i_n_dims_u-1] == 1 );

  int8_t l_gemm_type_s = i_types_s[i_n_dims_s-1];
  int8_t l_gemm_type_t = i_types_t[i_n_dims_t-1];
  int8_t l_gemm_type_u = i_types_u[i_n_dims_u-1];

  libxsmm_blasint l_gemm_m = 0;
  libxsmm_blasint l_gemm_n = 0;
  libxsmm_blasint l_gemm_k = 0;

  libxsmm_blasint l_gemm_lda = 0;
  libxsmm_blasint l_gemm_ldb = 0;
  libxsmm_blasint l_gemm_ldc = 0;

  libxsmm_bitfield l_gemm_flags = 0;
  libxsmm_bitfield l_gemm_prefetch_flags = 0;

  if(    l_gemm_type_s == 0
      && l_gemm_type_t == 0 ) {
    // A is column-major, B is row-major
    l_gemm_m = i_sizes_s[i_n_dims_s-1];
    l_gemm_n = i_sizes_t[i_n_dims_t-1];
    l_gemm_k = i_sizes_t[i_n_dims_t-2];

    l_gemm_lda = l_gemm_m;
    l_gemm_ldb = l_gemm_n;
    l_gemm_ldc = l_gemm_m;

    l_gemm_flags = LIBXSMM_GEMM_FLAGS('N', 'T');

    // TODO: row-major C
    assert( l_gemm_type_u == 0 );
  }
  else if(    l_gemm_type_s == 0
           && l_gemm_type_t == 1 ) {
    // A and B are column-major
    l_gemm_m = i_sizes_s[i_n_dims_s-1];
    l_gemm_n = i_sizes_t[i_n_dims_t-2];
    l_gemm_k = i_sizes_t[i_n_dims_t-1];

    l_gemm_lda = l_gemm_m;
    l_gemm_ldb = l_gemm_k;
    l_gemm_ldc = l_gemm_m;

    l_gemm_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

    // TODO: row-major C
    assert( l_gemm_type_u == 0 );
  }
  else if(    l_gemm_type_s == 1
           && l_gemm_type_t == 0 ) {
    // A and B are row-major
    l_gemm_m = i_sizes_s[i_n_dims_s-2];
    l_gemm_n = i_sizes_t[i_n_dims_t-1];
    l_gemm_k = i_sizes_s[i_n_dims_s-1];

    l_gemm_lda = l_gemm_k;
    l_gemm_ldb = l_gemm_n;
    l_gemm_ldc = l_gemm_m;

    l_gemm_flags = LIBXSMM_GEMM_FLAGS('T', 'T');

    // TODO: row-major C
    assert( l_gemm_type_u == 0 );
  }
  else if(    l_gemm_type_s == 1
           && l_gemm_type_t == 1 ) {
    // A is row-major, B is column-major
    l_gemm_m = i_sizes_s[i_n_dims_s-2];
    l_gemm_n = i_sizes_t[i_n_dims_t-2];
    l_gemm_k = i_sizes_s[i_n_dims_s-1];

    l_gemm_lda = l_gemm_k;
    l_gemm_ldb = l_gemm_k;
    l_gemm_ldc = l_gemm_m;

    l_gemm_flags = LIBXSMM_GEMM_FLAGS('T', 'N');

    // TODO: row-major C
    assert( l_gemm_type_u == 0 );
  }
  else {
    assert( false );
  }

  l_gemm_flags |= LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;

  libxsmm_gemm_shape l_gemm_shape = libxsmm_create_gemm_shape( l_gemm_m,
                                                               l_gemm_n,
                                                               l_gemm_k,
                                                               l_gemm_lda,
                                                               l_gemm_ldb,
                                                               l_gemm_ldc,
                                                               LIBXSMM_DATATYPE_F32,
                                                               LIBXSMM_DATATYPE_F32,
                                                               LIBXSMM_DATATYPE_F32,
                                                               LIBXSMM_DATATYPE_F32 );

  libxsmm_gemmfunction l_gemm = libxsmm_dispatch_gemm_v2( l_gemm_shape,
                                                          l_gemm_flags,
                                                          l_gemm_prefetch_flags );

  // configuration of the M loops
  int64_t l_m_loops_sizes[m_max_loops]     = { 0 };
  int64_t l_m_loops_strides_s[m_max_loops] = { 0 };
  int64_t l_m_loops_strides_u[m_max_loops] = { 0 };

  int64_t l_num_m_loops = loop_configs( i_n_dims_s,
                                        i_n_dims_u,
                                        0,
                                        0,
                                        i_types_s,
                                        i_types_u,
                                        i_sizes_s,
                                        i_strides_s,
                                        i_strides_u,
                                        l_m_loops_sizes,
                                        l_m_loops_strides_s,
                                        l_m_loops_strides_u );

  // configuration of the N loops
  int64_t l_n_loops_sizes[m_max_loops]     = { 0 };
  int64_t l_n_loops_strides_t[m_max_loops] = { 0 };
  int64_t l_n_loops_strides_u[m_max_loops] = { 0 };

  int64_t l_num_n_loops = loop_configs( i_n_dims_t,
                                        i_n_dims_u,
                                        0,
                                        1,
                                        i_types_t,
                                        i_types_u,
                                        i_sizes_t,
                                        i_strides_t,
                                        i_strides_u,
                                        l_n_loops_sizes,
                                        l_n_loops_strides_t,
                                        l_n_loops_strides_u );

  // configuration of the K loops
  int64_t l_k_loops_sizes[m_max_loops]     = { 0 };
  int64_t l_k_loops_strides_s[m_max_loops] = { 0 };
  int64_t l_k_loops_strides_t[m_max_loops] = { 0 };

  int64_t l_num_k_loops = loop_configs( i_n_dims_s,
                                        i_n_dims_t,
                                        1,
                                        1,
                                        i_types_s,
                                        i_types_t,
                                        i_sizes_s,
                                        i_strides_s,
                                        i_strides_t,
                                        l_k_loops_sizes,
                                        l_k_loops_strides_s,
                                        l_k_loops_strides_t );

  // TODO: add batch (B) loops

  int64_t l_m_loops_ctrs[m_max_loops] = { 0 };
  int64_t l_n_loops_ctrs[m_max_loops] = { 0 };
  int64_t l_k_loops_ctrs[m_max_loops] = { 0 };
  int64_t l_b_loops_ctrs[m_max_loops] = { 0 };

  while( true ) {
    int64_t l_offset_s = 0;
    int64_t l_offset_t = 0;
    int64_t l_offset_u = 0;
    for( int64_t l_loop_id_m = 0; l_loop_id_m < l_num_m_loops; l_loop_id_m++ ) {
      l_offset_s += l_m_loops_ctrs[l_loop_id_m] * l_m_loops_strides_s[l_loop_id_m];
      l_offset_u += l_m_loops_ctrs[l_loop_id_m] * l_m_loops_strides_u[l_loop_id_m];
    }
    for( int64_t l_loop_id_n = 0; l_loop_id_n < l_num_n_loops; l_loop_id_n++ ) {
      l_offset_t += l_n_loops_ctrs[l_loop_id_n] * l_n_loops_strides_t[l_loop_id_n];
      l_offset_u += l_n_loops_ctrs[l_loop_id_n] * l_n_loops_strides_u[l_loop_id_n];
    }
    for( int64_t l_loop_id_k = 0; l_loop_id_k < l_num_k_loops; l_loop_id_k++ ) {
      l_offset_s += l_k_loops_ctrs[l_loop_id_k] * l_k_loops_strides_s[l_loop_id_k];
      l_offset_t += l_k_loops_ctrs[l_loop_id_k] * l_k_loops_strides_t[l_loop_id_k];
    }

    libxsmm_gemm_param l_param;
    l_param.a.primary = (char *) i_s + l_offset_s * l_dtype_size;
    l_param.b.primary = (char *) i_t + l_offset_t * l_dtype_size;
    l_param.c.primary = (char *) o_u + l_offset_u * l_dtype_size;

    l_gemm( &l_param );

    bool l_advance_n = advance_loop( l_num_k_loops-2,
                                     l_k_loops_sizes,
                                     l_k_loops_ctrs );

    if( l_advance_n ) {
      bool l_advance_m = advance_loop( l_num_n_loops-2,
                                       l_n_loops_sizes,
                                       l_n_loops_ctrs );

      if( l_advance_m ) {
        bool l_finished = advance_loop( l_num_m_loops-2,
                                        l_m_loops_sizes,
                                        l_m_loops_ctrs );
        if( l_finished ) break;
      }
    }
  }

}