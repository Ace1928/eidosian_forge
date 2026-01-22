from typing import Dict, Tuple
from torch.distributed.checkpoint.metadata import (
from ._traverse import (
def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
    new_fqn = '.'.join(map(str, path))
    if new_fqn in flattened:
        raise ValueError(f'duplicated flatten key {new_fqn}')
    flattened[new_fqn] = value
    mappings[new_fqn] = path