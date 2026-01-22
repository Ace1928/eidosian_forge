import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def pformat_latex_qtree(self):
    """
        Returns a representation of the tree compatible with the
        LaTeX qtree package. This consists of the string ``\\Tree``
        followed by the tree represented in bracketed notation.

        For example, the following result was generated from a parse tree of
        the sentence ``The announcement astounded us``::

          \\Tree [.I'' [.N'' [.D The ] [.N' [.N announcement ] ] ]
              [.I' [.V'' [.V' [.V astounded ] [.N'' [.N' [.N us ] ] ] ] ] ] ]

        See https://www.ling.upenn.edu/advice/latex.html for the LaTeX
        style file for the qtree package.

        :return: A latex qtree representation of this tree.
        :rtype: str
        """
    reserved_chars = re.compile('([#\\$%&~_\\{\\}])')
    pformat = self.pformat(indent=6, nodesep='', parens=('[.', ' ]'))
    return '\\Tree ' + re.sub(reserved_chars, '\\\\\\1', pformat)