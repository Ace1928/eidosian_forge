from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def contract_path(*operands, **kwargs):
    """
    Find a contraction order 'path', without performing the contraction.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    optimize : str, list or bool, optional (default: ``auto``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - ``'optimal'`` An algorithm that explores all possible ways of
          contracting the listed tensors. Scales factorially with the number of
          terms in the contraction.
        - ``'branch-all'`` An algorithm like optimal but that restricts itself
          to searching 'likely' paths. Still scales factorially.
        - ``'branch-2'`` An even more restricted version of 'branch-all' that
          only searches the best two options at each step. Scales exponentially
          with the number of terms in the contraction.
        - ``'greedy'`` An algorithm that heuristically chooses the best pair
          contraction at each step.
        - ``'auto'`` Choose the best of the above algorithms whilst aiming to
          keep the path finding time below 1ms.

    use_blas : bool
        Use BLAS functions or not
    memory_limit : int, optional (default: None)
        Maximum number of elements allowed in intermediate arrays.
    shapes : bool, optional
        Whether ``contract_path`` should assume arrays (the default) or array
        shapes have been supplied.

    Returns
    -------
    path : list of tuples
        The einsum path
    PathInfo : str
        A printable object containing various information about the path found.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the end of
    the contraction list.

    Examples
    --------

    We can begin with a chain dot example. In this case, it is optimal to
    contract the b and c tensors represented by the first element of the path (1,
    2). The resulting tensor is added to the end of the contraction and the
    remaining contraction, ``(0, 1)``, is then executed.

    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = opt_einsum.contract_path('ij,jk,kl->il', a, b, c)
    >>> print(path_info[0])
    [(1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = oe.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)

    >>> print(path_info[0])
    [(0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """
    unknown_kwargs = set(kwargs) - _VALID_CONTRACT_KWARGS
    if len(unknown_kwargs):
        raise TypeError('einsum_path: Did not understand the following kwargs: {}'.format(unknown_kwargs))
    path_type = kwargs.pop('optimize', 'auto')
    memory_limit = kwargs.pop('memory_limit', None)
    shapes = kwargs.pop('shapes', False)
    einsum_call_arg = kwargs.pop('einsum_call', False)
    use_blas = kwargs.pop('use_blas', True)
    input_subscripts, output_subscript, operands = parser.parse_einsum_input(operands)
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    if shapes:
        input_shps = operands
    else:
        input_shps = [x.shape for x in operands]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))
    size_dict = {}
    for tnum, term in enumerate(input_list):
        sh = input_shps[tnum]
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript '{}' does not contain the correct number of indices for operand {}.".format(input_list[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])
            if char in size_dict:
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError("Size of label '{}' for operand {} ({}) does not match previous terms ({}).".format(char, tnum, size_dict[char], dim))
            else:
                size_dict[char] = dim
    size_list = [helpers.compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]]
    memory_arg = _choose_memory_arg(memory_limit, size_list)
    num_ops = len(input_list)
    inner_product = sum((len(x) for x in input_sets)) - len(indices) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)
    if not isinstance(path_type, (str, paths.PathOptimizer)):
        path = path_type
    elif num_ops <= 2:
        path = [tuple(range(num_ops))]
    elif isinstance(path_type, paths.PathOptimizer):
        path = path_type(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = paths.get_path_fn(path_type)
        path = path_optimizer(input_sets, output_set, size_dict, memory_arg)
    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []
    for cnum, contract_inds in enumerate(path):
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))
        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple
        cost = helpers.flop_count(idx_contract, idx_removed, len(contract_inds), size_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))
        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shps.pop(x) for x in contract_inds]
        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, out_inds, idx_removed, tmp_shapes)
        else:
            do_blas = False
        if cnum - len(path) == -1:
            idx_result = output_subscript
        else:
            all_input_inds = ''.join(tmp_inputs)
            idx_result = ''.join(sorted(out_inds, key=all_input_inds.find))
        shp_result = parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)
        input_list.append(idx_result)
        input_shps.append(shp_result)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result
        if len(input_list) <= 20:
            remaining = tuple(input_list)
        else:
            remaining = None
        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)
    opt_cost = sum(cost_list)
    if einsum_call_arg:
        return (operands, contraction_list)
    path_print = PathInfo(contraction_list, input_subscripts, output_subscript, indices, path, scale_list, naive_cost, opt_cost, size_list, size_dict)
    return (path, path_print)