import collections
import re
from tensorflow.python.util import tf_inspect
def get_strict_predicate_restore(registered_name):
    """Returns if the registered restore can be ignored if the predicate fails."""
    return _saver_registry.name_lookup(registered_name)[2]