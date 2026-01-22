from functools import partial
import numpy as np
from . import _catboost
def set_hints(self, **hints):
    for hint_key, hint_value in hints.items():
        if isinstance(hint_value, bool):
            hints[hint_key] = str(hint_value).lower()
    setattr(self, 'hints', '|'.join(['{}~{}'.format(hint_key, hint_value) for hint_key, hint_value in hints.items()]))
    if 'hints' not in self._params:
        self._params.append('hints')
    return self