from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array
@property
def found_inf(self):
    return self._found_inf