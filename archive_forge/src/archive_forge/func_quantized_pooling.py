from ._internal import NDArrayBase
from ..base import _Null
def quantized_pooling(data=None, min_data=None, max_data=None, kernel=_Null, pool_type=_Null, global_pool=_Null, cudnn_off=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, p_value=_Null, count_include_pad=_Null, layout=_Null, out=None, name=None, **kwargs):
    """Pooling operator for input and output data type of int8.
    The input and output data comes with min and max thresholds for quantizing
    the float32 data into int8.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.
        This operator only supports `pool_type` of `avg` or `max`.

    Defined in ../src/operator/quantization/quantized_pooling.cc:L186

    Parameters
    ----------
    data : NDArray
        Input data.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    kernel : Shape(tuple), optional, default=[]
        Pooling kernel size: (y, x) or (d, y, x)
    pool_type : {'avg', 'lp', 'max', 'sum'},optional, default='max'
        Pooling type to be applied.
    global_pool : boolean, optional, default=0
        Ignore kernel size, do global pooling based on current input feature map. 
    cudnn_off : boolean, optional, default=0
        Turn off cudnn pooling and use MXNet pooling operator. 
    pooling_convention : {'full', 'same', 'valid'},optional, default='valid'
        Pooling convention to be applied.
    stride : Shape(tuple), optional, default=[]
        Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
    p_value : int or None, optional, default='None'
        Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.
    count_include_pad : boolean or None, optional, default=None
        Only used for AvgPool, specify whether to count padding elements for averagecalculation. For example, with a 5*5 kernel on a 3*3 corner of a image,the sum of the 9 valid elements will be divided by 25 if this is set to true,or it will be divided by 9 if this is set to false. Defaults to true.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC', 'NWC'},optional, default='None'
        Set layout for input and output. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)