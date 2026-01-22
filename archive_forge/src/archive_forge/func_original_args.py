import collections
from tensorflow.python.util.tf_export import tf_export
@property
def original_args(self):
    """A `SessionRunArgs` object holding the original arguments of `run()`.

    If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
    field is equal to SessionRunArgs(a, b).

    Returns:
     A `SessionRunArgs` object
    """
    return self._original_args