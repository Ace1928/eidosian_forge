from ._internal import NDArrayBase
from ..base import _Null
def quantized_act(data=None, min_data=None, max_data=None, act_type=_Null, out=None, name=None, **kwargs):
    """Activation operator for input and output data type of int8.
    The input and output data comes with min and max thresholds for quantizing
    the float32 data into int8.

    .. Note::
         This operator only supports forward propogation. DO NOT use it in training.
         This operator only supports `relu`

    Defined in ../src/operator/quantization/quantized_activation.cc:L90

    Parameters
    ----------
    data : NDArray
        Input data.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    act_type : {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required
        Activation function to be applied.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)