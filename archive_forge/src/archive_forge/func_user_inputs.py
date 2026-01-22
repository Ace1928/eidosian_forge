import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def user_inputs(self) -> Collection[str]:
    return tuple((s.arg.name for s in self.input_specs if s.kind == InputKind.USER_INPUT and isinstance(s.arg, TensorArgument)))