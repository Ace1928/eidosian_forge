import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def gen_differentiable_outputs(fn: NativeFunctionWithDifferentiabilityInfo, key: str='Default') -> List[DifferentiableOutput]:
    f = fn.func
    info = fn.info[key] if fn.info else None
    outputs: List[DifferentiableOutput] = [DifferentiableOutput(name=name, type=ret.type, cpp_type=cpp.return_type(ret, symint=True).cpp_type()) for name, ret in zip(cpp.return_names(f), f.func.returns)]
    output_differentiability = info.output_differentiability if info else None
    if output_differentiability is not None:
        if len(output_differentiability) != len(outputs):
            raise RuntimeError(f'The length of output_differentiability ({len(output_differentiability)}), does not match the number of outputs ({len(outputs)}).')
        differentiable_outputs: List[DifferentiableOutput] = []
        if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
            raise RuntimeError("output_differentiability=False for inplace operation (version_counter won't get updated)")
        for differentiable, output in zip(output_differentiability, outputs):
            if differentiable:
                differentiable_outputs.append(output)
        return differentiable_outputs
    candidate_differentiable_outputs = list(filter(lambda r: is_differentiable(r.name, r.type, info), outputs))
    if uses_single_grad(info):
        return candidate_differentiable_outputs[:1]
    else:
        return candidate_differentiable_outputs