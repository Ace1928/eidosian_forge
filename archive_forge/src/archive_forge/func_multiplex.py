def multiplex(*brules):
    """ Multiplex many branching rules into one """

    def multiplex_brl(expr):
        seen = set()
        for brl in brules:
            for nexpr in brl(expr):
                if nexpr not in seen:
                    seen.add(nexpr)
                    yield nexpr
    return multiplex_brl