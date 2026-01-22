from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def replaceSingleToken(self, token, text):
    self.replace(self.DEFAULT_PROGRAM_NAME, token.tokenIndex, token.tokenIndex, text)