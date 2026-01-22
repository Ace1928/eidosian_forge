from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_dict(self):
    self.pop('integertype')
    self.push(ps_dict({}))