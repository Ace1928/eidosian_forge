from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getarrdims(a, var, verbose=0):
    ret = {}
    if isstring(var) and (not isarray(var)):
        ret['size'] = getstrlength(var)
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isscalar(var):
        ret['size'] = '1'
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isarray(var):
        dim = copy.copy(var['dimension'])
        ret['size'] = '*'.join(dim)
        try:
            ret['size'] = repr(eval(ret['size']))
        except Exception:
            pass
        ret['dims'] = ','.join(dim)
        ret['rank'] = repr(len(dim))
        ret['rank*[-1]'] = repr(len(dim) * [-1])[1:-1]
        for i in range(len(dim)):
            v = []
            if dim[i] in depargs:
                v = [dim[i]]
            else:
                for va in depargs:
                    if re.match('.*?\\b%s\\b.*' % va, dim[i]):
                        v.append(va)
            for va in v:
                if depargs.index(va) > depargs.index(a):
                    dim[i] = '*'
                    break
        ret['setdims'], i = ('', -1)
        for d in dim:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['setdims'] = '%s#varname#_Dims[%d]=%s,' % (ret['setdims'], i, d)
        if ret['setdims']:
            ret['setdims'] = ret['setdims'][:-1]
        ret['cbsetdims'], i = ('', -1)
        for d in var['dimension']:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (ret['cbsetdims'], i, d)
            elif isintent_in(var):
                outmess('getarrdims:warning: assumed shape array, using 0 instead of %r\n' % d)
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (ret['cbsetdims'], i, 0)
            elif verbose:
                errmess('getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n' % (repr(a), repr(d)))
        if ret['cbsetdims']:
            ret['cbsetdims'] = ret['cbsetdims'][:-1]
    return ret