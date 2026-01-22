from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
def parseStream(self, options, inStream):
    lexer = self.lexerClass(inStream)
    tokenStream = antlr3.CommonTokenStream(lexer)
    parser = self.parserClass(tokenStream)
    result = getattr(parser, options.parserRule)()
    if result is not None:
        assert hasattr(result, 'tree'), 'Parser did not return an AST'
        nodeStream = antlr3.tree.CommonTreeNodeStream(result.tree)
        nodeStream.setTokenStream(tokenStream)
        walker = self.walkerClass(nodeStream)
        result = getattr(walker, options.walkerRule)()
        if result is not None:
            if hasattr(result, 'tree'):
                self.writeln(options, result.tree.toStringTree())
            else:
                self.writeln(options, repr(result))