import numpy as np
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def get_np_dtype(x: TensorType) -> np.dtype:
    """Returns the NumPy dtype of the given tensor or array."""
    if torch and isinstance(x, torch.Tensor):
        return torch_to_numpy_dtype_dict[x.dtype]
    if tf and isinstance(x, tf.Tensor):
        return tf_to_numpy_dtype_dict[x.dtype]
    elif isinstance(x, np.ndarray):
        return x.dtype
    else:
        raise TypeError('Unsupported type: {}'.format(type(x)))