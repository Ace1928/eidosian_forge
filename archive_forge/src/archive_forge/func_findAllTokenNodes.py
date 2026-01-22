from io import StringIO
from antlr4.Token import Token
from antlr4.Utils import escapeWhitespace
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode, Tree, ParseTree
@classmethod
def findAllTokenNodes(cls, t: ParseTree, ttype: int):
    return cls.findAllNodes(t, ttype, True)