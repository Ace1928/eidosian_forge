from functools import partial
@property
def is_resolved(self):
    return self._values is None