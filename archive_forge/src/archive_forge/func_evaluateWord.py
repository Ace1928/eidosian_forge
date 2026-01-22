from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def evaluateWord(self, argument):
    return self.GetWord(argument[0])