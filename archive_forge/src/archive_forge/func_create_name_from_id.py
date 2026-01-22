import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
@staticmethod
def create_name_from_id(name, id, offset=None, max_count_digits=4):
    if offset is not None:
        name = '{name}{:0>{max_count_digits}}_offset{offset}'.format(id, name=name, max_count_digits=max_count_digits, offset=offset)
    else:
        name = '{name}{:0>{max_count_digits}}'.format(id, name=name, max_count_digits=max_count_digits)
    return name