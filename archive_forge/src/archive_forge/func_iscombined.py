from inspect import isclass
def iscombined(self):
    return any((expr.iscombined() for expr in self.exprs))