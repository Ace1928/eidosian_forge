import operator
import torch
from . import _dtypes_impl
def typecast_tensor(t, target_dtype, casting):
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    can_cast = _dtypes_impl.can_cast_impl
    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(f"Cannot cast array data from {t.dtype} to {target_dtype} according to the rule '{casting}'")
    return cast_if_needed(t, target_dtype)