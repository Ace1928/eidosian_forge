from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_noaccess(self):
    obj = self.pop()
    if obj.access < 3:
        obj.access = 3
    self.push(obj)