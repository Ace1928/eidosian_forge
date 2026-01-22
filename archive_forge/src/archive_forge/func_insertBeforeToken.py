from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def insertBeforeToken(self, token, text, program_name=DEFAULT_PROGRAM_NAME):
    self.insertBefore(program_name, token.tokenIndex, text)