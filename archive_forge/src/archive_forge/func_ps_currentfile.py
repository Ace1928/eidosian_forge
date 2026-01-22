from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_currentfile(self):
    self.push(ps_file(self.tokenizer))