from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_get(self):
    obj1 = self.pop()
    if obj1.value == 'Encoding':
        pass
    obj2 = self.pop('arraytype', 'dicttype', 'stringtype', 'proceduretype', 'fonttype')
    tp = obj2.type
    if tp in ('arraytype', 'proceduretype'):
        self.push(obj2.value[obj1.value])
    elif tp in ('dicttype', 'fonttype'):
        self.push(obj2.value[obj1.value])
    elif tp == 'stringtype':
        self.push(ps_integer(ord(obj2.value[obj1.value])))
    else:
        assert False, "shouldn't get here"