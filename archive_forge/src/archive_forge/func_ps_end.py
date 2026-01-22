from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_end(self):
    if len(self.dictstack) > 2:
        del self.dictstack[-1]
    else:
        raise RuntimeError('dictstack underflow')