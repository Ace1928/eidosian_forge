from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
Scope to prevent any update to Keras Variables.

    The values of variables to be used inside the scope
    should be passed via the `state_mapping` argument, a
    list of tuples `(k, v)` where `k` is a `KerasVariable`
    and `v` is the intended value for this variable
    (a backend tensor).

    Updated values can be collected on scope exit via
    `value = scope.get_current_value(variable)`. No updates
    will be applied in-place to any variables for the duration
    of the scope.

    Example:

    ```python
    state_mapping = [(k, ops.ones(k.shape, k.dtype)) for k in model.weights]
    with keras.StatelessScope(state_mapping) as scope:
        outputs = model.some_function(inputs)

    # All model variables remain unchanged. Their new values can be
    # collected via:
    for k in model.weights:
        new_value = scope.get_current_value(k)
        print(f"New value for {k}: {new_value})
    ```
    