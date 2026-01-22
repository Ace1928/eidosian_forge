import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def lp_pooling(attrs, inputs, proto_obj):
    """LP Pooling"""
    p_value = attrs.get('p', 2)
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape': 'kernel', 'strides': 'stride', 'pads': 'pad'})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['p'])
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'pooling_convention': 'valid', 'p_value': p_value})
    new_op = translation_utils._fix_pooling('lp', inputs, new_attrs)
    return (new_op, new_attrs, inputs)