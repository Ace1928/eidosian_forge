from ._internal import NDArrayBase
from ..base import _Null
def random_pdf_negative_binomial(sample=None, k=None, p=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of samples of
    negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).

    *k* and *p* must have the same shape, which must match the leftmost subshape
    of *sample*.  That is, *sample* can have the same shape as *k* and *p*, in which
    case the output contains one density per distribution, or *sample* can be a tensor
    of tensors with that shape, in which case the output is a tensor of densities such that
    the densities at index *i* in the output are given by the samples at index *i* in *sample*
    parameterized by the values of *k* and *p* at index *i*.

    Examples::

        random_pdf_negative_binomial(sample=[[1,2,3,4]], k=[1], p=a[0.5]) =
            [[0.25, 0.125, 0.0625, 0.03125]]

        # Note that k may be real-valued
        sample = [[1,2,3,4],
                  [1,2,3,4]]
        random_pdf_negative_binomial(sample=sample, k=[1, 1.5], p=[0.5, 0.5]) =
            [[0.25,       0.125,      0.0625,     0.03125   ],
             [0.26516506, 0.16572815, 0.09667476, 0.05437956]]


    Defined in ../src/operator/random/pdf_op.cc:L309

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    k : NDArray
        Limits of unsuccessful experiments.
    is_log : boolean, optional, default=0
        If set, compute the density of the log-probability instead of the probability.
    p : NDArray
        Failure probabilities in each experiment.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)