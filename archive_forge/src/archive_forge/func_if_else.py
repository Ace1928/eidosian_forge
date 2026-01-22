import contextlib
import functools
from llvmlite.ir import instructions, types, values
@contextlib.contextmanager
def if_else(self, pred, likely=None):
    """
        A context manager which sets up two conditional basic blocks based
        on the given predicate (a i1 value).
        A tuple of context managers is yield'ed.  Each context manager
        acts as a if_then() block.
        *likely* has the same meaning as in if_then().

        Typical use::
            with builder.if_else(pred) as (then, otherwise):
                with then:
                    # emit instructions for when the predicate is true
                with otherwise:
                    # emit instructions for when the predicate is false
        """
    bb = self.basic_block
    bbif = self.append_basic_block(name=_label_suffix(bb.name, '.if'))
    bbelse = self.append_basic_block(name=_label_suffix(bb.name, '.else'))
    bbend = self.append_basic_block(name=_label_suffix(bb.name, '.endif'))
    br = self.cbranch(pred, bbif, bbelse)
    if likely is not None:
        br.set_weights([99, 1] if likely else [1, 99])
    then = self._branch_helper(bbif, bbend)
    otherwise = self._branch_helper(bbelse, bbend)
    yield (then, otherwise)
    self.position_at_end(bbend)