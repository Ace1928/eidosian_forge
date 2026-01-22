from io import StringIO
from antlr4.Token import Token
from antlr4.Utils import escapeWhitespace
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode, Tree, ParseTree
@classmethod
def findAllNodes(cls, t: ParseTree, index: int, findTokens: bool):
    nodes = []
    cls._findAllNodes(t, index, findTokens, nodes)
    return nodes