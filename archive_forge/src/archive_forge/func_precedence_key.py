import functools
import re
import warnings
@property
def precedence_key(self):
    return self._sort_precedence_key