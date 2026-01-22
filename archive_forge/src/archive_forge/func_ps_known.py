from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_known(self):
    key = self.pop()
    d = self.pop('dicttype', 'fonttype')
    self.push(ps_boolean(key.value in d.value))