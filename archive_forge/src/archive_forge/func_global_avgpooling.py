import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def global_avgpooling(attrs, inputs, proto_obj):
    """Performs avg pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True, 'kernel': (1, 1), 'pool_type': 'avg'})
    return ('Pooling', new_attrs, inputs)