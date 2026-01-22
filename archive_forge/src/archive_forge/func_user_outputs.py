import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def user_outputs(self) -> Collection[str]:
    return tuple((s.arg.name for s in self.output_specs if s.kind == OutputKind.USER_OUTPUT and isinstance(s.arg, TensorArgument)))