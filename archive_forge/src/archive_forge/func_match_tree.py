import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def match_tree(self, tree, rulename):
    """Match the elements of `tree` to the symbols of rule `rulename`.

        Parameters:
            tree (Tree): the tree node to match
            rulename (str): The expected full rule name (including template args)

        Returns:
            Tree: an unreduced tree that matches `rulename`

        Raises:
            UnexpectedToken: If no match was found.

        Note:
            It's the callers' responsibility match the tree recursively.
        """
    if rulename:
        name, _args = parse_rulename(rulename)
        assert tree.data == name
    else:
        rulename = tree.data
    try:
        parser = self._parser_cache[rulename]
    except KeyError:
        rules = self.rules + _best_rules_from_group(self.rules_for_root[rulename])
        callbacks = {rule: rule.alias for rule in rules}
        conf = ParserConf(rules, callbacks, [rulename])
        parser = earley.Parser(self.parser.lexer_conf, conf, _match, resolve_ambiguity=True)
        self._parser_cache[rulename] = parser
    unreduced_tree = parser.parse(ChildrenLexer(tree.children), rulename)
    assert unreduced_tree.data == rulename
    return unreduced_tree