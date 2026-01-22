from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def strategy_supports_no_merge_call():
    """Returns if the current Strategy can operate in pure replica context."""
    if not distribute_lib.has_strategy():
        return True
    strategy = distribute_lib.get_strategy()
    return not strategy.extended._use_merge_call()