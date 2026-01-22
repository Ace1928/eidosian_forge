from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
def bind_global_function(self, fobj, ftype, args, kws={}):
    """Binds a global function to a variable.

        Parameters
        ----------
        fobj : object
            The function to be bound.
        ftype : types.Type
        args : Sequence[types.Type]
        kws : Mapping[str, types.Type]

        Returns
        -------
        callable: _CallableNode
        """
    loc = self._loc
    varname = f'{fobj.__name__}_func'
    gvname = f'{fobj.__name__}'
    func_sig = self._typingctx.resolve_function_type(ftype, args, kws)
    func_var = self.assign(rhs=ir.Global(gvname, fobj, loc=loc), typ=ftype, name=varname)
    return _CallableNode(func=func_var, sig=func_sig)