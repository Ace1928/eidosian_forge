from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_if(self):
    proc = self.pop('proceduretype')
    if self.pop('booleantype').value:
        self.call_procedure(proc)