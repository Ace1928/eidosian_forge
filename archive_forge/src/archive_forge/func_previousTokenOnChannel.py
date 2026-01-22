from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def previousTokenOnChannel(self, i: int, channel: int):
    while i >= 0 and self.tokens[i].channel != channel:
        i -= 1
    return i