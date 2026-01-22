from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def freepres(module):
    """
        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a
        submodule of ``F``, and ``Q`` a submodule of ``S``, such that
        ``module = S/Q``, and ``c`` is a conversion function.
        """
    if isinstance(module, FreeModule):
        return (module, module, module.submodule(), lambda x: module.convert(x))
    if isinstance(module, QuotientModule):
        return (module.base, module.base, module.killed_module, lambda x: module.convert(x).data)
    if isinstance(module, SubQuotientModule):
        return (module.base.container, module.base, module.killed_module, lambda x: module.container.convert(x).data)
    return (module.container, module, module.submodule(), lambda x: module.container.convert(x))