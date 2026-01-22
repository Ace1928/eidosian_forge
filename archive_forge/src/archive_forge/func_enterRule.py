from antlr4 import *
from io import StringIO
import sys
def enterRule(self, listener: ParseTreeListener):
    if hasattr(listener, 'enterIndexing'):
        listener.enterIndexing(self)