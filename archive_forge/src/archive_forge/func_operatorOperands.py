from pyparsing import Word, nums, alphas, Combine, oneOf, \
def operatorOperands(tokenlist):
    """generator to extract operators and operands in pairs"""
    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break