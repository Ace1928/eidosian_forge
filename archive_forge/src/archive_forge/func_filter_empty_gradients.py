from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def filter_empty_gradients(grads_and_vars):
    """Filter out `(grad, var)` pairs that have a gradient equal to `None`."""
    grads_and_vars = tuple(grads_and_vars)
    if not grads_and_vars:
        return grads_and_vars
    filtered = []
    vars_with_empty_grads = []
    for grad, var in grads_and_vars:
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append((grad, var))
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError('No gradients provided for any variable: %s.' % ([v.name for _, v in grads_and_vars],))
    if vars_with_empty_grads:
        logging.warning('Gradients do not exist for variables %s when minimizing the loss.', [v.name for v in vars_with_empty_grads])
    return filtered