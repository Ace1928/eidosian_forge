import copy
from .. import Options
def restore_Options(backup):
    no_value = object()
    for name, orig_value in backup.items():
        if getattr(Options, name, no_value) != orig_value:
            setattr(Options, name, orig_value)
    for name in vars(Options).keys():
        if name not in backup:
            delattr(Options, name)