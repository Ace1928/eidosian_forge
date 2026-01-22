from ._internal import NDArrayBase
from ..base import _Null
def quantized_conv(data=None, weight=None, bias=None, min_data=None, max_data=None, min_weight=None, max_weight=None, min_bias=None, max_bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, out=None, name=None, **kwargs):
    """Convolution operator for input, weight and bias data type of int8,
    and accumulates in type int32 for the output. For each argument, two more arguments of type
    float32 must be provided representing the thresholds of quantizing argument from data
    type float32 to int8. The final outputs contain the convolution result in int32, and min
    and max thresholds representing the threholds for quantizing the float32 output into int32.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in ../src/operator/quantization/quantized_conv.cc:L187

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
    kernel : Shape(tuple), required
        Convolution kernel size: (w,), (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
    num_filter : int (non-negative), required
        Convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages. When CUDNN is not used, it determines the effective batch size of the convolution kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is used.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are only supported on GPU.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)