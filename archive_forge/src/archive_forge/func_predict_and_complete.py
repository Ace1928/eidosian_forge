from typing import TYPE_CHECKING, Callable, Optional, List, Any
from collections import deque
from ..lexer import Token
from ..tree import Tree
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger, OrderedSet
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item
from .earley_forest import ForestSumVisitor, SymbolNode, StableSymbolNode, TokenNode, ForestToParseTree
def predict_and_complete(self, i, to_scan, columns, transitives):
    """The core Earley Predictor and Completer.

        At each stage of the input, we handling any completed items (things
        that matched on the last cycle) and use those to predict what should
        come next in the input stream. The completions and any predicted
        non-terminals are recursively processed until we reach a set of,
        which can be added to the scan list for the next scanner cycle."""
    node_cache = {}
    held_completions = {}
    column = columns[i]
    items = deque(column)
    while items:
        item = items.pop()
        if item.is_complete:
            if item.node is None:
                label = (item.s, item.start, i)
                item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                item.node.add_family(item.s, item.rule, item.start, None, None)
            if item.rule.origin in transitives[item.start]:
                transitive = transitives[item.start][item.s]
                if transitive.previous in transitives[transitive.column]:
                    root_transitive = transitives[transitive.column][transitive.previous]
                else:
                    root_transitive = transitive
                new_item = Item(transitive.rule, transitive.ptr, transitive.start)
                label = (root_transitive.s, root_transitive.start, i)
                new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                new_item.node.add_path(root_transitive, item.node)
                if new_item.expect in self.TERMINALS:
                    to_scan.add(new_item)
                elif new_item not in column:
                    column.add(new_item)
                    items.append(new_item)
            else:
                is_empty_item = item.start == i
                if is_empty_item:
                    held_completions[item.rule.origin] = item.node
                originators = [originator for originator in columns[item.start] if originator.expect is not None and originator.expect == item.s]
                for originator in originators:
                    new_item = originator.advance()
                    label = (new_item.s, originator.start, i)
                    new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                    new_item.node.add_family(new_item.s, new_item.rule, i, originator.node, item.node)
                    if new_item.expect in self.TERMINALS:
                        to_scan.add(new_item)
                    elif new_item not in column:
                        column.add(new_item)
                        items.append(new_item)
        elif item.expect in self.NON_TERMINALS:
            new_items = []
            for rule in self.predictions[item.expect]:
                new_item = Item(rule, 0, i)
                new_items.append(new_item)
            if item.expect in held_completions:
                new_item = item.advance()
                label = (new_item.s, item.start, i)
                new_item.node = node_cache[label] if label in node_cache else node_cache.setdefault(label, self.SymbolNode(*label))
                new_item.node.add_family(new_item.s, new_item.rule, new_item.start, item.node, held_completions[item.expect])
                new_items.append(new_item)
            for new_item in new_items:
                if new_item.expect in self.TERMINALS:
                    to_scan.add(new_item)
                elif new_item not in column:
                    column.add(new_item)
                    items.append(new_item)