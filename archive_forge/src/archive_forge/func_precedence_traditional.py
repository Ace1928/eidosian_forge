def precedence_traditional(item):
    """Returns the precedence of a given object according to the
    traditional rules of mathematics.

    This is the precedence for the LaTeX and pretty printer.
    """
    from sympy.core.expr import UnevaluatedExpr
    if isinstance(item, UnevaluatedExpr):
        return precedence_traditional(item.args[0])
    n = item.__class__.__name__
    if n in PRECEDENCE_TRADITIONAL:
        return PRECEDENCE_TRADITIONAL[n]
    return precedence(item)