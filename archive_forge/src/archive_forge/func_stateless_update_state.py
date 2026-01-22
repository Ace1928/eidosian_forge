from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
from keras.src.utils.tracking import Tracker
def stateless_update_state(self, metric_variables, *args, **kwargs):
    if len(metric_variables) != len(self.variables):
        raise ValueError(f'Argument `metric_variables` must be a list of tensors corresponding 1:1 to {self.__class__.__name__}().variables. Received list with length {len(metric_variables)}, but expected {len(self.variables)} variables.')
    mapping = list(zip(self.variables, metric_variables))
    with backend.StatelessScope(state_mapping=mapping) as scope:
        self.update_state(*args, **kwargs)
    metric_variables = []
    for v in self.variables:
        new_v = scope.get_current_value(v)
        if new_v is not None:
            metric_variables.append(new_v)
        else:
            metric_variables.append(v)
    return metric_variables