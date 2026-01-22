from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def evaluateWordWildcard(self, argument):
    return self.GetWordWildcard(argument[0])