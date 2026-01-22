import numpy
from cupy import _core
from cupyx.jit import _interface
from cupyx.jit import _cuda_types

        Args:
            pyfunc (callable): The target python function.
            otypes (str or list of dtypes, optional): The output data type.
            doc (str or None): The docstring for the function.
            excluded: Currently not supported.
            cache: Currently Ignored.
            signature: Currently not supported.
        