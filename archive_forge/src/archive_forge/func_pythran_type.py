from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_type(Ty, ptype='ndarray'):
    if Ty.is_buffer:
        ndim, dtype = (Ty.ndim, Ty.dtype)
        if isinstance(dtype, CStructOrUnionType):
            ctype = dtype.cname
        elif isinstance(dtype, CType):
            ctype = dtype.sign_and_name()
        elif isinstance(dtype, CTypedefType):
            ctype = dtype.typedef_cname
        else:
            raise ValueError('unsupported type %s!' % dtype)
        if pythran_is_pre_0_9:
            return 'pythonic::types::%s<%s,%d>' % (ptype, ctype, ndim)
        else:
            return 'pythonic::types::%s<%s,pythonic::types::pshape<%s>>' % (ptype, ctype, ','.join(('long',) * ndim))
    if Ty.is_pythran_expr:
        return Ty.pythran_type
    if Ty.is_numeric:
        return Ty.sign_and_name()
    raise ValueError('unsupported pythran type %s (%s)' % (Ty, type(Ty)))