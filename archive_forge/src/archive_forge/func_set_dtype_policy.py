from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
@keras_export(['keras.config.set_dtype_policy', 'keras.mixed_precision.set_dtype_policy', 'keras.mixed_precision.set_global_policy'])
def set_dtype_policy(policy):
    """Sets the default dtype policy globally.

    Example:

    >>> keras.config.set_dtype_policy("mixed_float16")
    """
    if not isinstance(policy, DTypePolicy):
        if isinstance(policy, str):
            policy = DTypePolicy(policy)
        else:
            raise ValueError(f"Invalid `policy` argument. Expected the string name of a policy (such as 'mixed_float16') or a `DTypePolicy` instance. Received: policy={policy} (of type {type(policy)})")
    global_state.set_global_attribute('dtype_policy', policy)