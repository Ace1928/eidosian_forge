from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_eexec(self):
    f = self.pop('filetype').value
    f.starteexec()