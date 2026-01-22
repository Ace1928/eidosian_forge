from ._internal import NDArrayBase
from ..base import _Null
def quantized_fully_connected(data=None, weight=None, bias=None, min_data=None, max_data=None, min_weight=None, max_weight=None, min_bias=None, max_bias=None, num_hidden=_Null, no_bias=_Null, flatten=_Null, out=None, name=None, **kwargs):
    """Fully Connected operator for input, weight and bias data type of int8,
    and accumulates in type int32 for the output. For each argument, two more arguments of type
    float32 must be provided representing the thresholds of quantizing argument from data
    type float32 to int8. The final outputs contain the convolution result in int32, and min
    and max thresholds representing the threholds for quantizing the float32 output into int32.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in ../src/operator/quantization/quantized_fully_connected.cc:L312

    Parameters
    ----------
    data : NDArray
        Input data.
    weight : NDArray
        weight.
    bias : NDArray
        bias.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    min_weight : NDArray
        Minimum value of weight.
    max_weight : NDArray
        Maximum value of weight.
    min_bias : NDArray
        Minimum value of bias.
    max_bias : NDArray
        Maximum value of bias.
    num_hidden : int, required
        Number of hidden nodes of the output.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    flatten : boolean, optional, default=1
        Whether to collapse all but the first axis of the input data tensor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)