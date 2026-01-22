from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_index(self):
    n = self.pop('integertype').value
    if n < 0:
        raise RuntimeError('index may not be negative')
    self.push(self.stack[-1 - n])