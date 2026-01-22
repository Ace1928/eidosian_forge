from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def evaluateAnd(self, argument):
    return self.evaluate(argument[0]).intersection(self.evaluate(argument[1]))