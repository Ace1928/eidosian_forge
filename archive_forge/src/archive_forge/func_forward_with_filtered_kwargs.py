import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def forward_with_filtered_kwargs(self, *args, **kwargs):
    signature = dict(inspect.signature(self.forward).parameters)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
    return self(*args, **filtered_kwargs)