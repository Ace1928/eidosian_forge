import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def end_statement(self, stmt):
    """Marks the end of a statement.

    Args:
      stmt: Hashable, a key by which the statement can be identified in the
        CFG's stmt_prev and stmt_next attributes; must match a key previously
        passed to begin_statement.
    """
    self.active_stmts.remove(stmt)