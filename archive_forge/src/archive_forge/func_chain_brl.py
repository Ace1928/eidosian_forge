def chain_brl(expr):
    if not brules:
        yield expr
        return
    head, tail = (brules[0], brules[1:])
    for nexpr in head(expr):
        yield from chain(*tail)(nexpr)