def conditioned_brl(expr):
    if cond(expr):
        yield from brule(expr)
    else:
        pass