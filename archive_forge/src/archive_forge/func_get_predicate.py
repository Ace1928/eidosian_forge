import collections
import re
from tensorflow.python.util import tf_inspect
def get_predicate(self, registered_name):
    try:
        return self._registered_predicates[registered_name]
    except KeyError:
        raise LookupError(f"The {self.name} registry does not have name '{registered_name}' registered.")