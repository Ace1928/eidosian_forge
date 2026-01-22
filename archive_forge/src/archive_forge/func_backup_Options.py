import copy
from .. import Options
def backup_Options():
    backup = {}
    for name, value in vars(Options).items():
        if name == '_directive_defaults':
            value = copy.deepcopy(value)
        backup[name] = value
    return backup