from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_exch(self):
    if len(self.stack) < 2:
        raise RuntimeError('stack underflow')
    obj1 = self.pop()
    obj2 = self.pop()
    self.push(obj1)
    self.push(obj2)