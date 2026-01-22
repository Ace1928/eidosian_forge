from pythran.passmanager import Transformation
import gast as ast
def getid(node):
    if isinstance(node, ast.Attribute):
        return (getid(node.value), node.attr)
    if isinstance(node, ast.Name):
        return node.id
    return node