from contextlib import contextmanager
from copy import copy
def set_upward(self, key, value):
    """
        Set a variable in one of the higher contexts if it exists there,
        otherwise in the current context.
        """
    context = self.dicts[-1]
    for d in reversed(self.dicts):
        if key in d:
            context = d
            break
    context[key] = value