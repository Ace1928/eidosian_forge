from antlr4 import *
from io import StringIO
import sys
def exitRule(self, listener: ParseTreeListener):
    if hasattr(listener, 'exitIndexing'):
        listener.exitIndexing(self)