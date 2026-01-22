from ._internal import NDArrayBase
from ..base import _Null
def nag_mom_update(weight=None, grad=None, mom=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, out=None, name=None, **kwargs):
    """Update function for Nesterov Accelerated Gradient( NAG) optimizer.
    It updates the weights using the following formula,

    .. math::
      v_t = \\gamma v_{t-1} + \\eta * \\nabla J(W_{t-1} - \\gamma v_{t-1})\\\\
      W_t = W_{t-1} - v_t

    Where 
    :math:`\\eta` is the learning rate of the optimizer
    :math:`\\gamma` is the decay rate of the momentum estimate
    :math:`\\v_t` is the update vector at time step `t`
    :math:`\\W_t` is the weight vector at time step `t`



    Defined in ../src/operator/optimizer_op.cc:L725

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    mom : NDArray
        Momentum
    lr : float, required
        Learning rate
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
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