import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def sort_overloads(grouped_overloads: Sequence[PythonSignatureGroup], *, symint: bool=True) -> Sequence[PythonSignatureGroup]:

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return str(t1) == 'Scalar' and str(t2) == 'Tensor' or (str(t1) == 'Scalar?' and str(t2) == 'Tensor?') or ('Dimname' in str(t1) and 'Dimname' not in str(t2)) or (str(t1) == 'int[]' and (str(t2) == 'int' or str(t2) == 'int?')) or (str(t1) == 'Tensor[]' and str(t2).find('[]') != -1) or (str(t1) == 'SymInt[]' and str(t2) == 'int[]') or ((str(t1) == 'SymInt' or str(t1) == 'int') and str(t2) == 'Tensor')

    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        args1, args2 = (s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True))
        if len(args1) != len(args2):
            return False
        equal = all((arg1.type == arg2.type for arg1, arg2 in zip(args1, args2)))
        smaller_or_equal = all((str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type) for arg1, arg2 in zip(args1, args2)))
        return smaller_or_equal and (not equal)
    grouped_overloads = sorted(grouped_overloads, key=lambda x: x.signature.signature_str(symint=symint))
    larger_than: Dict[int, Set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)
    if not larger_than:
        return list(grouped_overloads)
    N = len(grouped_overloads)
    sorted_ids: List[int] = list(filter(lambda x: x not in larger_than, range(N)))
    for idx in range(N):
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)
    return [grouped_overloads[x] for x in sorted_ids]