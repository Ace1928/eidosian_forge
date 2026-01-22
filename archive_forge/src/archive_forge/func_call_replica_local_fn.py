from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables
def call_replica_local_fn(fn, *args, **kwargs):
    """Call a function that uses replica-local variables.

  This function correctly handles calling `fn` in a cross-replica
  context.

  Args:
    fn: The function to call.
    *args: Positional arguments to the `fn`.
    **kwargs: Keyword argument to `fn`.

  Returns:
    The result of calling `fn`.
  """
    strategy = None
    if 'strategy' in kwargs:
        strategy = kwargs.pop('strategy')
    elif distribute_lib.has_strategy():
        strategy = distribute_lib.get_strategy()
    is_tpu = backend.is_tpu_strategy(strategy)
    if not is_tpu and strategy and distribute_lib.in_cross_replica_context():
        with strategy.scope():
            return strategy.extended.call_for_each_replica(fn, args, kwargs)
    return fn(*args, **kwargs)