import jax
import numpy as np
from keras.src.utils import jax_utils
def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Note that this function can be used both in eager context, or within a
    jitted function.

    Args:
        tensor: `jax.Array` that need to be distributed.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed value.
    """
    if not isinstance(layout, jax.sharding.Sharding):
        layout = _to_jax_layout(layout)
    if jax_utils.is_in_jax_tracing_scope():
        return jax.lax.with_sharding_constraint(tensor, layout)
    if layout.is_fully_addressable:
        return jax.device_put(tensor, layout)
    else:
        mapping = layout.addressable_devices_indices_map(tensor.shape)
        local_values = jax.device_put([tensor[i] for i in mapping.values()], list(mapping.keys()))
        global_value = jax.make_array_from_single_device_arrays(tensor.shape, layout, local_values)
        return global_value