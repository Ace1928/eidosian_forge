import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def import_should_record_backprop():
    """Return should_record_backprop or an equivalent function from TensorFlow."""
    import tensorflow.python as tfpy
    if hasattr(tfpy.eager.tape, 'should_record_backprop'):
        from tensorflow.python.eager.tape import should_record_backprop
    elif hasattr(tfpy.eager.tape, 'should_record'):
        from tensorflow.python.eager.tape import should_record as should_record_backprop
    elif hasattr(tfpy.eager.record, 'should_record_backprop'):
        from tensorflow.python.eager.record import should_record_backprop
    else:
        raise ImportError('Cannot import should_record_backprop from TensorFlow.')
    return should_record_backprop