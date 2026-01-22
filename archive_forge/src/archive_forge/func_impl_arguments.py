from typing import List, Union
from torchgen.api import cpp
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
def impl_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if g.out.precomputed:
        non_out_args_replaced: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
        for a in g.out.func.arguments.non_out:
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                for replacement in g.out.precomputed.replace[a.name]:
                    non_out_args_replaced.append(replacement)
            else:
                non_out_args_replaced.append(a)
        args.extend(non_out_args_replaced)
        args.extend(g.out.precomputed.add)
    else:
        args.extend(g.out.func.arguments.non_out)
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]