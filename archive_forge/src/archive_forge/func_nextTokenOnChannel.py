from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def nextTokenOnChannel(self, i: int, channel: int):
    self.sync(i)
    if i >= len(self.tokens):
        return len(self.tokens) - 1
    token = self.tokens[i]
    while token.channel != channel:
        if token.type == Token.EOF:
            return i
        i += 1
        self.sync(i)
        token = self.tokens[i]
    return i