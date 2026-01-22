import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
def visit_node(self, node):
    prev_defs_out = self.out[node]
    defs_in = _NodeState()
    for n in node.prev:
        defs_in |= self.out[n]
    if anno.hasanno(node.ast_node, anno.Static.SCOPE):
        node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
        if node not in self.gen_map:
            node_symbols = {}
            newly_defined = (node_scope.bound | node_scope.globals) - node_scope.deleted
            for s in newly_defined:
                def_ = self._definition_factory()
                node_symbols[s] = def_
            for s, p in node_scope.params.items():
                def_ = self._definition_factory()
                def_.param_of = weakref.ref(p)
                node_symbols[s] = def_
            self.gen_map[node] = _NodeState(node_symbols)
        gen = self.gen_map[node]
        kill = node_scope.modified | node_scope.deleted
        defs_out = gen | defs_in - kill
        gen = self.gen_map[node]
        defs_out = gen | defs_in - kill
    else:
        assert self.can_ignore(node), (node.ast_node, node)
        defs_out = defs_in
    self.in_[node] = defs_in
    self.out[node] = defs_out
    return prev_defs_out != defs_out