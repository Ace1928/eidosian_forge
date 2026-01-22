from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def get_jump_target(self):
    assert self.is_jump
    if PYVERSION in ((3, 12),):
        if self.opcode in (dis.opmap[k] for k in ['JUMP_BACKWARD']):
            return self.offset - (self.arg - 1) * 2
    elif PYVERSION in ((3, 11),):
        if self.opcode in (dis.opmap[k] for k in ('JUMP_BACKWARD', 'POP_JUMP_BACKWARD_IF_TRUE', 'POP_JUMP_BACKWARD_IF_FALSE', 'POP_JUMP_BACKWARD_IF_NONE', 'POP_JUMP_BACKWARD_IF_NOT_NONE')):
            return self.offset - (self.arg - 1) * 2
    elif PYVERSION in ((3, 9), (3, 10)):
        pass
    else:
        raise NotImplementedError(PYVERSION)
    if PYVERSION in ((3, 10), (3, 11), (3, 12)):
        if self.opcode in JREL_OPS:
            return self.next + self.arg * 2
        else:
            assert self.opcode in JABS_OPS
            return self.arg * 2 - 2
    elif PYVERSION in ((3, 9),):
        if self.opcode in JREL_OPS:
            return self.next + self.arg
        else:
            assert self.opcode in JABS_OPS
            return self.arg
    else:
        raise NotImplementedError(PYVERSION)