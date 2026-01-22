import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def sample_multinomial(attrs, inputs, proto_obj):
    """Draw random samples from a multinomial distribution."""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError:
        raise ImportError('Onnx and protobuf need to be installed. ' + 'Instructions to install - https://github.com/onnx/onnx')
    new_attrs = translation_utils._remove_attributes(attrs, ['seed'])
    new_attrs = translation_utils._fix_attribute_names(new_attrs, {'sample_size': 'shape'})
    new_attrs['dtype'] = TENSOR_TYPE_TO_NP_TYPE[int(attrs.get('dtype', 6))]
    return ('sample_multinomial', new_attrs, inputs)