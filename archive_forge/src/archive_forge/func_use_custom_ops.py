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
def use_custom_ops(subscripts, *operands, output_type):
    x, y = (operands[0], operands[1])
    if subscripts == 'a,b->ab':
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=0)
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'ab,b->a':
        y = tf.expand_dims(y, axis=-1)
        result = tf.matmul(x, y, output_type=output_type)
        return tf.squeeze(result, axis=-1)
    elif subscripts == 'ab,bc->ac':
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'ab,cb->ac':
        y = tf.transpose(y, [1, 0])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abc,cd->abd':
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abc,cde->abde':
        _, b1, c1 = x.shape
        c2, d2, e2 = y.shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        y = tf.reshape(y, [c, -1])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.reshape(result, [-1, b, d, e])
    elif subscripts == 'abc,dc->abd':
        y = tf.transpose(y, [1, 0])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abc,dce->abde':
        _, b1, c1 = x.shape
        d2, c2, e2 = y.shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        y = tf.transpose(y, [1, 0, 2])
        y = tf.reshape(y, [c, -1])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.reshape(result, [-1, b, d, e])
    elif subscripts == 'abc,dec->abde':
        _, b1, c1 = x.shape
        d2, e2, c2 = y.shape
        b, c, d, e = (b1, c1 or c2, d2, e2)
        y = tf.transpose(y, [2, 0, 1])
        y = tf.reshape(y, [c, -1])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.reshape(result, [-1, b, d, e])
    elif subscripts == 'abcd,abde->abce':
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcd,abed->abce':
        y = tf.transpose(y, [0, 1, 3, 2])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcd,acbe->adbe':
        x = tf.transpose(x, [0, 1, 3, 2])
        y = tf.transpose(y, [0, 2, 1, 3])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.transpose(result, [0, 2, 1, 3])
    elif subscripts == 'abcd,adbe->acbe':
        y = tf.transpose(y, [0, 2, 1, 3])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.transpose(result, [0, 2, 1, 3])
    elif subscripts == 'abcd,aecd->acbe':
        x = tf.transpose(x, [0, 2, 1, 3])
        y = tf.transpose(y, [0, 2, 3, 1])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcd,aecd->aceb':
        x = tf.transpose(x, [0, 2, 1, 3])
        y = tf.transpose(y, [0, 2, 3, 1])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.transpose(result, [0, 1, 3, 2])
    elif subscripts == 'abcd,cde->abe':
        _, b1, c1, d1 = x.shape
        c2, d2, e2 = y.shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        x = tf.reshape(x, [-1, b, c * d])
        y = tf.reshape(y, [-1, e])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcd,ced->abe':
        _, b1, c1, d1 = x.shape
        c2, e2, d2 = y.shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        x = tf.reshape(x, [-1, b, c * d])
        y = tf.transpose(y, [0, 2, 1])
        y = tf.reshape(y, [-1, e])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcd,ecd->abe':
        _, b1, c1, d1 = x.shape
        e2, c2, d2 = y.shape
        b, c, d, e = (b1, c1 or c2, d1 or d2, e2)
        x = tf.reshape(x, [-1, b, c * d])
        y = tf.transpose(y, [1, 2, 0])
        y = tf.reshape(y, [-1, e])
        return tf.matmul(x, y, output_type=output_type)
    elif subscripts == 'abcde,aebf->adbcf':
        _, b1, c1, d1, e1 = x.shape
        _, e2, b2, f2 = y.shape
        b, c, d, e, f = (b1 or b2, c1, d1, e1 or e2, f2)
        x = tf.reshape(x, [-1, b, c * d, e])
        y = tf.transpose(y, [0, 2, 1, 3])
        result = tf.matmul(x, y, output_type=output_type)
        result = tf.reshape(result, [-1, b, c, d, f])
        return tf.transpose(result, [0, 3, 1, 2, 4])
    elif subscripts == 'abcde,afce->acdbf':
        _, b1, c1, d1, e1 = x.shape
        _, f2, c2, e2 = y.shape
        b, c, d, e, f = (b1, c1 or c2, d1, e1 or e2, f2)
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        x = tf.reshape(x, [-1, c, d * b, e])
        y = tf.transpose(y, [0, 2, 3, 1])
        result = tf.matmul(x, y, output_type=output_type)
        return tf.reshape(result, [-1, c, d, b, f])
    else:
        raise NotImplementedError