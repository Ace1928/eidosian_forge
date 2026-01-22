from ._internal import NDArrayBase
from ..base import _Null
def signsgd_update(weight=None, grad=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Update function for SignSGD optimizer.

    .. math::

     g_t = \\nabla J(W_{t-1})\\\\
     W_t = W_{t-1} - \\eta_t \\text{sign}(g_t)

    It updates the weights using::

     weight = weight - learning_rate * sign(gradient)

    .. note::
       - sparse ndarray not supported for this optimizer yet.


    Defined in ../src/operator/optimizer_op.cc:L62

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    lr : float, required
        Learning rate
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)