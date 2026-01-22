from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def evaluateOr(self, argument):
    return self.evaluate(argument[0]).union(self.evaluate(argument[1]))