def precedence_Rational(item):
    if item.p < 0:
        return PRECEDENCE['Add']
    return PRECEDENCE['Mul']