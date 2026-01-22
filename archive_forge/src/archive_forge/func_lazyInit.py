from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def lazyInit(self):
    if self.index == -1:
        self.setup()