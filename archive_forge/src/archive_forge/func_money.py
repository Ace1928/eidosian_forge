from .Format import Format, Group, Scheme, Sign, Symbol
def money(decimals, sign=Sign.default):
    return Format(group=Group.yes, precision=decimals, scheme=Scheme.fixed, sign=sign, symbol=Symbol.yes)