from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_dup(self):
    if not self.stack:
        raise RuntimeError('stack underflow')
    self.push(self.stack[-1])