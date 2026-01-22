import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def visit_Nonlocal(self, node):
    self._enter_scope(False)
    for name in node.names:
        qn = qual_names.QN(name)
        self.scope.read.add(qn)
        self.scope.bound.add(qn)
        self.scope.nonlocals.add(qn)
    self._exit_and_record_scope(node)
    return node