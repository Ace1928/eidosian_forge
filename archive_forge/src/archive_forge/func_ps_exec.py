from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_exec(self):
    obj = self.pop()
    if obj.type == 'proceduretype':
        self.call_procedure(obj)
    else:
        self.handle_object(obj)