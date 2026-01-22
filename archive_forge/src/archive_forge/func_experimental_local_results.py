from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export
def experimental_local_results(self, value):
    """Returns the list of all local per-replica values contained in `value`.

    In `CentralStorageStrategy` there is a single worker so the value returned
    will be all the values on that worker.

    Args:
      value: A value returned by `run()`, `extended.call_for_each_replica()`,
      or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    """
    return super(CentralStorageStrategy, self).experimental_local_results(value)