def precedence_FracElement(item):
    if item.denom == 1:
        return precedence_PolyElement(item.numer)
    else:
        return PRECEDENCE['Mul']