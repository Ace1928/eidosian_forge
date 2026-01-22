import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def lifted_tensor_constants(self) -> Collection[str]:
    return [s.target for s in self.input_specs if s.kind == InputKind.CONSTANT_TENSOR if isinstance(s.target, str)]