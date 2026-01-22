from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_cvn(self):
    self.push(ps_name(self.pop('stringtype').value))