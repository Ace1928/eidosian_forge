from ._internal import NDArrayBase
from ..base import _Null
fill target with zeros without default dtype

    Parameters
    ----------
    shape : Shape(tuple), optional, default=None
        The shape of the output
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : int, optional, default='-1'
        Target data type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    