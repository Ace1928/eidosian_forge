from torch.fx.graph_module import GraphModule
from typing import Any, Callable, Dict, List, Tuple, Type
import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> Dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    attrs_for_lowering: Dict[str, Any] = {}
    attrs_for_lowering['name'] = torch.typename(mod)
    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(f'Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, please upgrade the module_fetch_book, open an issue and @842974287 or report a bug to AIACC team directly.')
        for attr in param_to_fetch:
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        raise RuntimeError(f'{torch.typename(mod)} is not in the module_fetch_book yet, please add it to the module_fetch_book, open an issue and @842974287 or report a bug to AIACC team directly.')
    return attrs_for_lowering