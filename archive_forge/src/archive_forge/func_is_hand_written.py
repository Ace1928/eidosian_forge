from typing import Dict, Union
from torchgen.model import NativeFunctionsGroup, NativeFunctionsViewGroup
def is_hand_written(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> bool:
    name_base = func_name_base_str(g)
    return name_base in is_hand_written_ops_