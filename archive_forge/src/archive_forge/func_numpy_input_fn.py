from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
from six import string_types
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.inputs.queues import feeding_functions
@estimator_export(v1=['estimator.inputs.numpy_input_fn'])
def numpy_input_fn(x, y=None, batch_size=128, num_epochs=1, shuffle=None, queue_capacity=1000, num_threads=1):
    """Returns input function that would feed dict of numpy arrays into the model.

  This returns a function outputting `features` and `targets` based on the dict
  of numpy arrays. The dict `features` has the same keys as the `x`. The dict
  `targets` has the same keys as the `y` if `y` is a dict.

  Example:

  ```python
  age = np.arange(4) * 1.0
  height = np.arange(32, 36)
  x = {'age': age, 'height': height}
  y = np.arange(-32, -28)

  with tf.Session() as session:
    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
  ```

  Args:
    x: numpy array object or dict of numpy array objects. If an array, the array
      will be treated as a single feature.
    y: numpy array object or dict of numpy array object. `None` if absent.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
      time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing. In
      order to have predicted and repeatable order of reading and enqueueing,
      such as in prediction and evaluation mode, `num_threads` should be 1.

  Returns:
    Function, that has signature of ()->(dict of `features`, `targets`)

  Raises:
    ValueError: if the shape of `y` mismatches the shape of values in `x` (i.e.,
      values in `x` have same shape).
    ValueError: if duplicate keys are in both `x` and `y` when `y` is a dict.
    ValueError: if x or y is an empty dict.
    TypeError: `x` is not a dict or array.
    ValueError: if 'shuffle' is not provided or a bool.
  """
    if not isinstance(shuffle, bool):
        raise ValueError('shuffle must be provided and explicitly set as boolean (it is recommended to set it as True for training); got {}'.format(shuffle))

    def input_fn():
        """Numpy input function."""
        ordered_dict_data = _validate_and_convert_features(x)
        feature_keys = list(ordered_dict_data.keys())
        if y is None:
            target_keys = None
        elif isinstance(y, dict):
            if not y:
                raise ValueError('y cannot be empty dict, use None instead.')
            ordered_dict_y = collections.OrderedDict(sorted(y.items(), key=lambda t: t[0]))
            target_keys = list(ordered_dict_y.keys())
            duplicate_keys = set(feature_keys).intersection(set(target_keys))
            if duplicate_keys:
                raise ValueError('{} duplicate keys are found in both x and y: {}'.format(len(duplicate_keys), duplicate_keys))
            ordered_dict_data.update(ordered_dict_y)
        else:
            target_keys = _get_unique_target_key(ordered_dict_data)
            ordered_dict_data[target_keys] = y
        if len(set((v.shape[0] for v in ordered_dict_data.values()))) != 1:
            shape_dict_of_x = {k: ordered_dict_data[k].shape for k in feature_keys}
            if target_keys is None:
                shape_of_y = None
            elif isinstance(target_keys, string_types):
                shape_of_y = y.shape
            else:
                shape_of_y = {k: ordered_dict_data[k].shape for k in target_keys}
            raise ValueError('Length of tensors in x and y is mismatched. All elements in x and y must have the same length.\nShapes in x: {}\nShapes in y: {}\n'.format(shape_dict_of_x, shape_of_y))
        queue = feeding_functions._enqueue_data(ordered_dict_data, queue_capacity, shuffle=shuffle, num_threads=num_threads, enqueue_size=batch_size, num_epochs=num_epochs)
        batch = queue.dequeue_many(batch_size) if num_epochs is None else queue.dequeue_up_to(batch_size)
        if batch:
            batch.pop(0)
        if isinstance(x, np.ndarray):
            features = batch[0]
        else:
            features = dict(zip(feature_keys, batch[:len(feature_keys)]))
        if target_keys is None:
            return features
        elif isinstance(target_keys, string_types):
            target = batch[-1]
            return (features, target)
        else:
            target = dict(zip(target_keys, batch[-len(target_keys):]))
            return (features, target)
    return input_fn