import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def user_inputs_to_mutate(self) -> Mapping[str, str]:
    return {s.arg.name: s.target for s in self.output_specs if s.kind == OutputKind.USER_INPUT_MUTATION and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}