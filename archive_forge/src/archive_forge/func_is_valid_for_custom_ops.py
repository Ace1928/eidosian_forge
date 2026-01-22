import builtins
import collections
import functools
import math
import string
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
from keras.src.utils import tree
def is_valid_for_custom_ops(subscripts, *operands):
    if subscripts in ['a,b->ab', 'ab,b->a', 'ab,bc->ac', 'ab,cb->ac', 'abc,cd->abd', 'abc,dc->abd', 'abcd,abde->abce', 'abcd,abed->abce', 'abcd,acbe->adbe', 'abcd,adbe->acbe', 'abcd,aecd->acbe', 'abcd,aecd->aceb']:
        return True
    elif subscripts == 'abc,cde->abde':
        _, b1, c1 = operands[0].shape
        c2, d2, e2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abc,dce->abde':
        _, b1, c1 = operands[0].shape
        d2, c2, e2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abc,dec->abde':
        _, b1, c1 = operands[0].shape
        d2, e2, c2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abcd,cde->abe':
        _, b1, c1, d1 = operands[0].shape
        c2, d2, e2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abcd,ced->abe':
        _, b1, c1, d1 = operands[0].shape
        c2, e2, d2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abcd,ecd->abe':
        _, b1, c1, d1 = operands[0].shape
        e2, c2, d2 = operands[1].shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        if None in (b, c, d, e):
            return False
        return True
    elif subscripts == 'abcde,aebf->adbcf':
        _, b1, c1, d1, e1 = operands[0].shape
        _, e2, b2, f2 = operands[1].shape
        b, c, d, e, f = (b1 or b2, c1, d1, e1 or e2, f2)
        if None in (b, c, d, e, f):
            return False
        return True
    elif subscripts == 'abcde,afce->acdbf':
        _, b1, c1, d1, e1 = operands[0].shape
        _, f2, c2, e2 = operands[1].shape
        b, c, d, e, f = (b1, c1 or c2, d1, e1 or e2, f2)
        if None in (b, c, d, e, f):
            return False
        return True
    else:
        return False