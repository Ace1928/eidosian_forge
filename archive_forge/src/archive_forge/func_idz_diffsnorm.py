import scipy.linalg._interpolative as _id
import numpy as np
def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=20):
    """
    Estimate spectral norm of the difference of two complex matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the adjoint of the first matrix to a vector, with
        call signature `y = matveca(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matveca: function
    :param matveca2:
        Function to apply the adjoint of the second matrix to a vector, with
        call signature `y = matveca2(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matveca2: function
    :param matvec:
        Function to apply the first matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param matvec2:
        Function to apply the second matrix to a vector, with call signature
        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec2: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate of matrix difference.
    :rtype: float
    """
    return _id.idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its)