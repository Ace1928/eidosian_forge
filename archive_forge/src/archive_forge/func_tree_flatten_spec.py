from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional
from torch.utils._pytree import LeafSpec, PyTree, TreeSpec
def tree_flatten_spec(pytree: PyTree, spec: TreeSpec, exact_structural_match=False) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(f'{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make sure that any custom pytrees have been registered before loading it.')
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec)
    if exact_structural_match:
        flatten_fn_exact_match_spec = SUPPORTED_NODES_EXACT_MATCH[spec.type]
        if flatten_fn_exact_match_spec and (not flatten_fn_exact_match_spec(pytree, spec)):
            raise RuntimeError(f'Cannot flatten pytree {pytree}, given spec: {spec}')
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec, exact_structural_match)
        result += flat
    return result