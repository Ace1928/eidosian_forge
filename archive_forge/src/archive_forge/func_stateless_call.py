import collections
import inspect
import warnings
from functools import wraps
import tree
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import regularizers
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.distribution import distribution_lib
from keras.src.layers import input_spec
from keras.src.metrics.metric import Metric
from keras.src.ops.operation import Operation
from keras.src.utils import python_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils.shape_utils import map_shape_structure
@traceback_utils.filter_traceback
def stateless_call(self, trainable_variables, non_trainable_variables, *args, return_losses=False, **kwargs):
    """Call the layer without any side effects.

        Args:
            trainable_variables: List of trainable variables of the model.
            non_trainable_variables: List of non-trainable variables of the
                model.
            *args: Positional argumets to be passed to `call()`.
            return_losses: If `True`, `stateless_call()` will return the list of
                losses created during `call()` as part of its return values.
            **kwargs: Keyword arguments to be passed to `call()`.

        Returns:
            A tuple. By default, returns `(outputs, non_trainable_variables)`.
                If `return_losses = True`, then returns
                `(outputs, non_trainable_variables, losses)`.

        Note: `non_trainable_variables` include not only non-trainable weights
        such as `BatchNormalization` statistics, but also RNG seed state
        (if there are any random operations part of the layer, such as dropout),
        and `Metric` state (if there are any metrics attached to the layer).
        These are all elements of state of the layer.

        Example:

        ```python
        model = ...
        data = ...
        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        # Call the model with zero side effects
        outputs, non_trainable_variables = model.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
        )
        # Attach the updated state to the model
        # (until you do this, the model is still in its pre-call state).
        for ref_var, value in zip(
            model.non_trainable_variables, non_trainable_variables
        ):
            ref_var.assign(value)
        ```
        """
    self._check_super_called()
    if not self.built:
        raise ValueError(f'To call stateless_call, {self.__class__.__name__} must be built (i.e. its variables must have been already created). You can build it by calling it on some data.')
    if len(trainable_variables) != len(self.trainable_variables):
        raise ValueError(f'Argument `trainable_variables` must be a list of tensors corresponding 1:1 to {self.__class__.__name__}().trainable_variables. Received list with length {len(trainable_variables)}, but expected {len(self.trainable_variables)} variables.')
    if len(non_trainable_variables) != len(self.non_trainable_variables):
        raise ValueError(f'Argument `non_trainable_variables` must be a list of tensors corresponding 1:1 to {self.__class__.__name__}().non_trainable_variables. Received list with length {len(non_trainable_variables)}, but expected {len(self.non_trainable_variables)} variables.')
    trainable_mapping = zip(self.trainable_variables, trainable_variables)
    non_trainable_mapping = zip(self.non_trainable_variables, non_trainable_variables)
    mapping = list(trainable_mapping) + list(non_trainable_mapping)
    losses = None
    with backend.StatelessScope(state_mapping=mapping, collect_losses=return_losses) as scope:
        outputs = self.call(*args, **kwargs)
        if return_losses:
            losses = self.losses
    non_trainable_variables = []
    for v in self.non_trainable_variables:
        new_v = scope.get_current_value(v)
        if new_v is not None:
            non_trainable_variables.append(new_v)
        else:
            non_trainable_variables.append(v)
    if return_losses:
        return (outputs, non_trainable_variables, losses)
    return (outputs, non_trainable_variables)