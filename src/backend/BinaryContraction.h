#ifndef TPP_NETS_BACKEND_BINARY_CONTRACTION

#include <cstdint>

namespace tpp_nets {
  namespace backend {
    class BinaryContraction;
  }
}

class tpp_nets::backend::BinaryContraction {
    static constexpr int64_t m_max_loops = 25;

    /**
     * Filters an array based on the elements' type.
     *
     * Given an array with i_size element.
     * Each element has a type (i_types) and a attribute (i_attributes).
     *
     * The routine iterates from [0, i_size-1] through the array.
     * Whenever the the of an element matches i_type_filter,
     * the elements' attribute is copied over to o_attribute.
     *
     * Copy operations to o_attribute start at position 0.
     * The position is incremented after every copy operation.
     *
     * @param i_size size of the input array holding elements with a type and attribute.
     * @param i_type_filter type which is filtered.
     * @param i_types types of the array's elements.
     * @param i_attributes attributes of the array's elements.
     * @param o_attributes will be set to filtered attributes.
     * @return number copy operations, i.e., number of times the filtered type occurred in the input.
     **/
    static int64_t filter_attributes( int64_t         i_size,
                                      int8_t          i_type_filter,
                                      int8_t  const * i_types,
                                      int64_t const * i_attributes,
                                      int64_t       * o_attributes );
    /**
     * Derives the configuration of loops iterating over dimension in a binary tensor contraction.
     *
     * @param i_n_dims_a number of dimensions of A.
     * @param i_n_dims_b number of dimensions of B.
     * @param i_types_a types of A's dimensions.
     * @param i_types_b types of B's dimensions.
     * @param i_sizes_a sizes of A's dimensions.
     * @param i_strides_a strides of A's dimensions.
     * @param i_strides_b strides of B's dimensions.
     * @param o_loops_sizes will be set to sizes of the resulting loops.
     * @param o_loops_strides_a will be set to strides of the resulting loops w.r.t. A.
     * @param o_loops_strides_b will be set to strides of the resulting loops w.r.t. B.
     * @return number of loops.
     **/
    static int64_t loop_configs( int64_t         i_n_dims_a,
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
                                 int64_t       * o_loops_strides_b );

    /**
     * Advance the given loops if possible.
     * If not the loop counters are set to zero.
     *
     * @param i_num_loops number of loops.
     * @param i_sizes sizes of the loops.
     * @param io_counters loop counters.
     * @return true if the counter have been set to zero, false otherwise. 
     **/
    bool advance_loop( int64_t         i_num_loops,
                       int64_t const * i_sizes,
                       int64_t       * io_counters );

  public:
    /**
     * Performs a (generalized) tensordot operation using Tensor Processing Primitives.
     * S and T are the input tensors, U is the output tensors.
     * 
     * @param i_n_dims_s S's number of dimensions.
     * @param i_n_dims_t T's number of dimensions.
     * @param i_n_dims_u U's number of dimensions.
     * @param i_sizes_s sizes of S's dimensions.
     * @param i_sizes_t sizes of T's dimensions.
     * @param i_sizes_u sizes of U's dimensions.
     * @param i_types_s types of S's dimensions (0: M, 1: K, 2: B).
     * @param i_types_t types of T's dimensions (0: N, 1: K, 2: B).
     * @param i_types_u types of U's dimensions (0: M, 1: N, 2: B).
     * @param i_strides_s strides of S's dimensions.
     * @param i_strides_t strides of T's dimensions.
     * @param i_strides_u strides of U's dimensions.
     * @param i_s data pointer of S.
     * @param i_t data pointer of T.
     * @param o_u data pointer of U.
     **/
    void tppdot( int64_t         i_n_dims_s,
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
                 void          * o_u );
};

#endif