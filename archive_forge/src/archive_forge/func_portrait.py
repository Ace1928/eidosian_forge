from reportlab.lib.units import mm, inch
def portrait(pagesize):
    """Use this to get page orientation right"""
    a, b = pagesize
    if a >= b:
        return (b, a)
    else:
        return (a, b)