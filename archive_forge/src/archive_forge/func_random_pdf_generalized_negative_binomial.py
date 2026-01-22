from ._internal import NDArrayBase
from ..base import _Null
def random_pdf_generalized_negative_binomial(sample=None, mu=None, alpha=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of *sample* of
    generalized negative binomial distributions with parameters *mu* (mean)
    and *alpha* (dispersion).  This can be understood as a reparameterization of
    the negative binomial, where *k* = *1 / alpha* and *p* = *1 / (mu \\* alpha + 1)*.

    *mu* and *alpha* must have the same shape, which must match the leftmost subshape
    of *sample*.  That is, *sample* can have the same shape as *mu* and *alpha*, in which
    case the output contains one density per distribution, or *sample* can be a tensor
    of tensors with that shape, in which case the output is a tensor of densities such that
    the densities at index *i* in the output are given by the samples at index *i* in *sample*
    parameterized by the values of *mu* and *alpha* at index *i*.

    Examples::

        random_pdf_generalized_negative_binomial(sample=[[1, 2, 3, 4]], alpha=[1], mu=[1]) =
            [[0.25, 0.125, 0.0625, 0.03125]]

        sample = [[1,2,3,4],
                  [1,2,3,4]]
        random_pdf_generalized_negative_binomial(sample=sample, alpha=[1, 0.6666], mu=[1, 1.5]) =
            [[0.25,       0.125,      0.0625,     0.03125   ],
             [0.26517063, 0.16573331, 0.09667706, 0.05437994]]


    Defined in ../src/operator/random/pdf_op.cc:L313

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    mu : NDArray
        Means of the distributions.
    is_log : boolean, optional, default=0
        If set, compute the density of the log-probability instead of the probability.
    alpha : NDArray
        Alpha (dispersion) parameters of the distributions.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)