def precedence_Integer(item):
    if item.p < 0:
        return PRECEDENCE['Add']
    return PRECEDENCE['Atom']