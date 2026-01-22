from ._internal import NDArrayBase
from ..base import _Null
Converts an image NDArray of shape (H x W x C) or (N x H x W x C) 
    with values in the range [0, 255] to a tensor NDArray of shape (C x H x W) or (N x C x H x W)
    with values in the range [0, 1]

    Example:
        .. code-block:: python
            image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
            to_tensor(image)
                [[[ 0.85490197  0.72156864]
                  [ 0.09019608  0.74117649]
                  [ 0.61960787  0.92941177]
                  [ 0.96470588  0.1882353 ]]
                 [[ 0.6156863   0.73725492]
                  [ 0.46666667  0.98039216]
                  [ 0.44705883  0.45490196]
                  [ 0.01960784  0.8509804 ]]
                 [[ 0.39607844  0.03137255]
                  [ 0.72156864  0.52941179]
                  [ 0.16470589  0.7647059 ]
                  [ 0.05490196  0.70588237]]]
                 <NDArray 3x4x2 @cpu(0)>

            image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
            to_tensor(image)
                [[[[0.11764706 0.5803922 ]
                   [0.9411765  0.10588235]
                   [0.2627451  0.73333335]
                   [0.5647059  0.32156864]]
                  [[0.7176471  0.14117648]
                   [0.75686276 0.4117647 ]
                   [0.18431373 0.45490196]
                   [0.13333334 0.6156863 ]]
                  [[0.6392157  0.5372549 ]
                   [0.52156866 0.47058824]
                   [0.77254903 0.21568628]
                   [0.01568628 0.14901961]]]
                 [[[0.6117647  0.38431373]
                   [0.6784314  0.6117647 ]
                   [0.69411767 0.96862745]
                   [0.67058825 0.35686275]]
                  [[0.21960784 0.9411765 ]
                   [0.44705883 0.43529412]
                   [0.09803922 0.6666667 ]
                   [0.16862746 0.1254902 ]]
                  [[0.6156863  0.9019608 ]
                   [0.35686275 0.9019608 ]
                   [0.05882353 0.6509804 ]
                   [0.20784314 0.7490196 ]]]]
                <NDArray 2x3x4x2 @cpu(0)>


    Defined in ../src/operator/image/image_random.cc:L92

    Parameters
    ----------
    data : NDArray
        Input ndarray

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    