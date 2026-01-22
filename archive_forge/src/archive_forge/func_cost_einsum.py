import math
def cost_einsum(x):
    eq, *operands = x.args
    lhs = eq.split('->')[0]
    terms = lhs.split(',')
    size_dict = {ix: d for term, x in zip(terms, operands) for ix, d in zip(term, x.shape)}
    return math.prod(size_dict.values())