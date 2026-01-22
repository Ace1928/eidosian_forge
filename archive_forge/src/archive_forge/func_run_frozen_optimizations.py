from typing import List, Optional
import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule
def run_frozen_optimizations(mod, optimize_numerics: bool=True, preserved_methods: Optional[List[str]]=None):
    """
    Run a series of optimizations looking for patterns that occur in frozen graphs.

    The current set of optimizations includes:
        - Dropout Removal
        - Pretranspose Linear Layers
        - Concat Linear Layers with same input Tensor
        - Conv -> Batchnorm folding
        - Conv -> Add/Sub folding
        - Conv -> Mul/Div folding

    Args:
        mod (:class:`ScriptModule`): a frozen module to be optimized

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_close`
        when applied on a single transformation, however in a module where many transformations are applied
        the rtol or atol may no longer fall within the default `assert_close` tolerance. Conv -> Batchnorm folding,
        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.

    Returns:
        None

    Note:
        In rare occassions, this can result in slower execution.

    Example (Freezing a module with Conv->Batchnorm)
    .. code-block:: python
        import torch
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)
        # inspect frozen mod
        assert "batch_norm" in str(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        assert "batch_norm" not in str(frozen_mod.graph)

    """
    if mod._c._has_method('forward'):
        torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)
    if preserved_methods is None:
        preserved_methods = []
    for method in preserved_methods:
        torch._C._jit_pass_optimize_frozen_graph(mod.__getattr__(method).graph, optimize_numerics)