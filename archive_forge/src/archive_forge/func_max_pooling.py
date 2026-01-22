import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def max_pooling(attrs, inputs, proto_obj):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape': 'kernel', 'strides': 'stride', 'pads': 'pad'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'pooling_convention': 'valid'})
    new_op = translation_utils._fix_pooling('max', inputs, new_attrs)
    return (new_op, new_attrs, inputs)