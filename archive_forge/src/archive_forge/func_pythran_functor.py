from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_functor(func):
    func = np_func_to_list(func)
    submodules = '::'.join(func[:-1] + ['functor'])
    return 'pythonic::numpy::%s::%s' % (submodules, func[-1])