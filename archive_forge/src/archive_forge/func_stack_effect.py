import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def stack_effect(self, jump=None):
    if self._opcode < _opcode.HAVE_ARGUMENT:
        arg = None
    elif not isinstance(self._arg, int) or self._opcode in _opcode.hasconst:
        arg = 0
    else:
        arg = self._arg
    if sys.version_info < (3, 8):
        effect = _stack_effects.get(self._opcode, None)
        if effect is not None:
            return max(effect) if jump is None else effect[jump]
        return dis.stack_effect(self._opcode, arg)
    else:
        return dis.stack_effect(self._opcode, arg, jump=jump)