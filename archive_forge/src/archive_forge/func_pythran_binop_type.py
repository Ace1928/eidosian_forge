from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_binop_type(op, tA, tB):
    if op == '**':
        return 'decltype(pythonic::numpy::functor::power{}(std::declval<%s>(), std::declval<%s>()))' % (pythran_type(tA), pythran_type(tB))
    else:
        return 'decltype(std::declval<%s>() %s std::declval<%s>())' % (pythran_type(tA), op, pythran_type(tB))