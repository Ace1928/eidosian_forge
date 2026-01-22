from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_for(self):
    proc = self.pop('proceduretype')
    limit = self.pop('integertype', 'realtype').value
    increment = self.pop('integertype', 'realtype').value
    i = self.pop('integertype', 'realtype').value
    while 1:
        if increment > 0:
            if i > limit:
                break
        elif i < limit:
            break
        if type(i) == type(0.0):
            self.push(ps_real(i))
        else:
            self.push(ps_integer(i))
        self.call_procedure(proc)
        i = i + increment