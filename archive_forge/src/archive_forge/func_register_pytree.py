from typing import Callable, Tuple, Any
def register_pytree(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    """Register a type with all available pytree backends.

    Current backends is jax.
    Args:
        pytree_type (type): the type to register, such as ``qml.RX``
        flatten_fn (Callable): a function that splits an object into trainable leaves and hashable metadata.
        unflatten_fn (Callable): a function that reconstructs an object from its leaves and metadata.

    Returns:
        None

    Side Effects:
        ``pytree`` type becomes registered with available backends.

    """
    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)