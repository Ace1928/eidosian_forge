from ._internal import NDArrayBase
from ..base import _Null
def random_pdf_normal(sample=None, mu=None, sigma=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of *sample* of
    normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).

    *mu* and *sigma* must have the same shape, which must match the leftmost subshape
    of *sample*.  That is, *sample* can have the same shape as *mu* and *sigma*, in which
    case the output contains one density per distribution, or *sample* can be a tensor
    of tensors with that shape, in which case the output is a tensor of densities such that
    the densities at index *i* in the output are given by the samples at index *i* in *sample*
    parameterized by the values of *mu* and *sigma* at index *i*.

    Examples::

        sample = [[-2, -1, 0, 1, 2]]
        random_pdf_normal(sample=sample, mu=[0], sigma=[1]) =
            [[0.05399097, 0.24197073, 0.3989423, 0.24197073, 0.05399097]]

        random_pdf_normal(sample=sample*2, mu=[0,0], sigma=[1,2]) =
            [[0.05399097, 0.24197073, 0.3989423,  0.24197073, 0.05399097],
             [0.12098537, 0.17603266, 0.19947115, 0.17603266, 0.12098537]]


    Defined in ../src/operator/random/pdf_op.cc:L299

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    mu : NDArray
        Means of the distributions.
    is_log : boolean, optional, default=0
        If set, compute the density of the log-probability instead of the probability.
    sigma : NDArray
        Standard deviations of the distributions.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)