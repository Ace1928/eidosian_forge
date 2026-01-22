import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def reduce_l1(attrs, inputs, proto_obj):
    """Reduce input tensor by l1 normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes': 'axis'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'ord': 1})
    return ('norm', new_attrs, inputs)