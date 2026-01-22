from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_cvx(self):
    obj = self.pop()
    obj.literal = 0
    self.push(obj)