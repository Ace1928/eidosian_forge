import contextlib
import functools
from llvmlite.ir import instructions, types, values
@contextlib.contextmanager
def if_then(self, pred, likely=None):
    """
        A context manager which sets up a conditional basic block based
        on the given predicate (a i1 value).  If the conditional block
        is not explicitly terminated, a branch will be added to the next
        block.
        If *likely* is given, its boolean value indicates whether the
        predicate is likely to be true or not, and metadata is issued
        for LLVM's optimizers to account for that.
        """
    bb = self.basic_block
    bbif = self.append_basic_block(name=_label_suffix(bb.name, '.if'))
    bbend = self.append_basic_block(name=_label_suffix(bb.name, '.endif'))
    br = self.cbranch(pred, bbif, bbend)
    if likely is not None:
        br.set_weights([99, 1] if likely else [1, 99])
    with self._branch_helper(bbif, bbend):
        yield bbend
    self.position_at_end(bbend)