from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast
def update_name(self, node):
    if isinstance(node, ast.Name) and node.id in self.argsid:
        if node.id not in self.renaming:
            new_name = node.id
            while new_name in self.identifiers:
                new_name = new_name + '_'
            self.renaming[node.id] = new_name