from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def replaceRange(self, from_idx, to_idx, text):
    self.replace(self.DEFAULT_PROGRAM_NAME, from_idx, to_idx, text)