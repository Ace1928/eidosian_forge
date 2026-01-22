from weakref import ref as weakref_ref
def removeSymbol(self, obj):
    symb = self.byObject.pop(id(obj))
    self.bySymbol.pop(symb)