from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_put(self):
    obj1 = self.pop()
    obj2 = self.pop()
    obj3 = self.pop('arraytype', 'dicttype', 'stringtype', 'proceduretype')
    tp = obj3.type
    if tp == 'arraytype' or tp == 'proceduretype':
        obj3.value[obj2.value] = obj1
    elif tp == 'dicttype':
        obj3.value[obj2.value] = obj1
    elif tp == 'stringtype':
        index = obj2.value
        obj3.value = obj3.value[:index] + chr(obj1.value) + obj3.value[index + 1:]