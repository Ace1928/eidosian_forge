from ._internal import NDArrayBase
from ..base import _Null
def lamb_update_phase2(weight=None, g=None, r1=None, r2=None, lr=_Null, lower_bound=_Null, upper_bound=_Null, out=None, name=None, **kwargs):
    """Phase II of lamb update it performs the following operations and updates grad.

    Link to paper: https://arxiv.org/pdf/1904.00962.pdf

    .. math::
        \\begin{gather*}
        if (lower_bound >= 0)
        then
             r1 = max(r1, lower_bound)
        if (upper_bound >= 0)
        then
             r1 = max(r1, upper_bound)

        if (r1 == 0 or r2 == 0)
        then
             lr = lr
        else
             lr = lr * (r1/r2)
        weight = weight - lr * g
        \\end{gather*}



    Defined in ../src/operator/optimizer_op.cc:L991

    Parameters
    ----------
    weight : NDArray
        Weight
    g : NDArray
        Output of lamb_update_phase 1
    r1 : NDArray
        r1
    r2 : NDArray
        r2
    lr : float, required
        Learning rate
    lower_bound : float, optional, default=-1
        Lower limit of norm of weight. If lower_bound <= 0, Lower limit is not set
    upper_bound : float, optional, default=-1
        Upper limit of norm of weight. If upper_bound <= 0, Upper limit is not set

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)