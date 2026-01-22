from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_type(self):
    obj = self.pop()
    self.push(ps_string(obj.type))