from collections import deque
from ..tree import Tree
from ..visitors import Transformer_InPlace, v_args
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item, TransitiveItem
from .earley_forest import ForestSumVisitor, SymbolNode, ForestToParseTree
@v_args(meta=True)
def drv(self, children, meta):
    return self.postprocess[meta.rule](children)