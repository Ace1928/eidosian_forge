from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def deleteToken(self, token):
    self.delete(self.DEFAULT_PROGRAM_NAME, token, token)