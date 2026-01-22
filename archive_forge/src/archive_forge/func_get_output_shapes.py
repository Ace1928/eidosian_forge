import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def get_output_shapes(self):
    """Get the shapes of the outputs."""
    outputs = self.execs[0].outputs
    shapes = [out.shape for out in outputs]
    concat_shapes = []
    for key, the_shape, axis in zip(self.symbol.list_outputs(), shapes, self.output_layouts):
        the_shape = list(the_shape)
        if axis >= 0:
            the_shape[axis] = self.batch_size
        concat_shapes.append((key, tuple(the_shape)))
    return concat_shapes