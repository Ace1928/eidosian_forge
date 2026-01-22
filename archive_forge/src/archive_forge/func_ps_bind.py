from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_bind(self):
    proc = self.pop('proceduretype')
    self.proc_bind(proc)
    self.push(proc)