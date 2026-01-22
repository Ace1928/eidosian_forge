from ._internal import NDArrayBase
from ..base import _Null
def make_loss(data=None, out=None, name=None, **kwargs):
    """Make your own loss function in network construction.

    This operator accepts a customized loss function symbol as a terminal loss and
    the symbol should be an operator with no backward dependency.
    The output of this function is the gradient of loss with respect to the input data.

    For example, if you are a making a cross entropy loss function. Assume ``out`` is the
    predicted output and ``label`` is the true label, then the cross entropy can be defined as::

      cross_entropy = label * log(out) + (1 - label) * log(1 - out)
      loss = make_loss(cross_entropy)

    We will need to use ``make_loss`` when we are creating our own loss function or we want to
    combine multiple loss functions. Also we may want to stop some variables' gradients
    from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.

    The storage type of ``make_loss`` output depends upon the input storage type:

       - make_loss(default) = default
       - make_loss(row_sparse) = row_sparse



    Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L358

    Parameters
    ----------
    data : NDArray
        The input array.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)