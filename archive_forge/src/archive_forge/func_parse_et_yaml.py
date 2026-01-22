from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
from torchgen.utils import NamespaceHelper
def parse_et_yaml(path: str, tags_yaml_path: str, ignore_keys: Optional[Set[DispatchKey]]=None, skip_native_fns_gen: bool=False) -> Tuple[List[NativeFunction], Dict[OperatorName, Dict[str, Any]]]:
    """Parse native_functions.yaml into NativeFunctions and an Operator Indexed Dict
    of fields to persist from native_functions.yaml to functions.yaml
    """
    with open(path) as f:
        es = yaml.load(f, Loader=LineLoader)
    et_kernel = extract_kernel_fields(es)
    strip_et_fields(es)
    native_yaml = parse_native_yaml(path, tags_yaml_path, ignore_keys, skip_native_fns_gen=skip_native_fns_gen, loaded_yaml=es)
    return (native_yaml.native_functions, et_kernel)