import types
from contextlib import contextmanager
from torch.backends import (
def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False