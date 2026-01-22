import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos
def lamba_check(self, fn_ast_node):
    if isinstance(fn_ast_node, gast.Lambda):
        return True
    return False