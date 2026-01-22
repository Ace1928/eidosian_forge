import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def global_maxpooling(attrs, inputs, proto_obj):
    """Performs max pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True, 'kernel': (1, 1), 'pool_type': 'max'})
    return ('Pooling', new_attrs, inputs)