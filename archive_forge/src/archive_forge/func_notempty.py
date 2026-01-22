def notempty(brule):

    def notempty_brl(expr):
        yielded = False
        for nexpr in brule(expr):
            yielded = True
            yield nexpr
        if not yielded:
            yield expr
    return notempty_brl