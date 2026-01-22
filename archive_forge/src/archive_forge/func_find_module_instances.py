from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union
from torch import Tensor, nn
def find_module_instances(module: nn.Module, search_class: Type[nn.Module]) -> List[Tuple[str, nn.Module]]:
    """
    Find all occurrences of a given search_class among the given Modules's
    children and return the corresponding paths in the same format as
    state_dicts.

    Usage::

        net = nn.Sequential(
            nn.Linear(1, 1),
            nn.ModuleDict({"ln": nn.LayerNorm(1), "linear": nn.Linear(1, 1)}),
            nn.LayerNorm(1)
        )

        >>> find_module_instances(net, nn.LayerNorm)
        [('1.ln.', LayerNorm((1,), eps=1e-05, elementwise_affine=True)), ('2.', LayerNorm((1,), eps=1e-05, elementwise_affine=True))]
        >>> find_module_instances(net, nn.Dropout)
        []
        >>> find_module_instances(net, nn.Sequential)
        [('', Sequential(
          (0): Linear(in_features=1, out_features=1, bias=True)
          (1): ModuleDict(
            (ln): LayerNorm((1,), eps=1e-05, elementwise_affine=True)
            (linear): Linear(in_features=1, out_features=1, bias=True)
          )
          (2): LayerNorm((1,), eps=1e-05, elementwise_affine=True)
        ))]
    """
    paths = []

    def add_paths_(module: nn.Module, prefix: str='') -> None:
        if isinstance(module, search_class):
            paths.append((prefix, module))
        for name, child in module.named_children():
            add_paths_(child, prefix + name + '.')
    add_paths_(module)
    return paths