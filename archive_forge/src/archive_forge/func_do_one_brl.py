def do_one_brl(expr):
    yielded = False
    for brl in brules:
        for nexpr in brl(expr):
            yielded = True
            yield nexpr
        if yielded:
            return