from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def insertAfterToken(self, token, text, program_name=DEFAULT_PROGRAM_NAME):
    self.insertAfter(token.tokenIndex, text, program_name)