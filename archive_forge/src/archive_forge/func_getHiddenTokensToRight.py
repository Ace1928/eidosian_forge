from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def getHiddenTokensToRight(self, tokenIndex: int, channel: int=-1):
    self.lazyInit()
    if tokenIndex < 0 or tokenIndex >= len(self.tokens):
        raise Exception(str(tokenIndex) + ' not in 0..' + str(len(self.tokens) - 1))
    from antlr4.Lexer import Lexer
    nextOnChannel = self.nextTokenOnChannel(tokenIndex + 1, Lexer.DEFAULT_TOKEN_CHANNEL)
    from_ = tokenIndex + 1
    to = len(self.tokens) - 1 if nextOnChannel == -1 else nextOnChannel
    return self.filterForChannel(from_, to, channel)