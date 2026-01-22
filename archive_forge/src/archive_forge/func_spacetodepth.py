import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def spacetodepth(attrs, inputs, proto_obj):
    """Rearranges blocks of spatial data into depth."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'blocksize': 'block_size'})
    return ('space_to_depth', new_attrs, inputs)