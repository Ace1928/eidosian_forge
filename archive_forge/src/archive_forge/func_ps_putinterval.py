from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_putinterval(self):
    obj1 = self.pop('arraytype', 'stringtype')
    obj2 = self.pop('integertype')
    obj3 = self.pop('arraytype', 'stringtype')
    tp = obj3.type
    if tp == 'arraytype':
        obj3.value[obj2.value:obj2.value + len(obj1.value)] = obj1.value
    elif tp == 'stringtype':
        newstr = obj3.value[:obj2.value]
        newstr = newstr + obj1.value
        newstr = newstr + obj3.value[obj2.value + len(obj1.value):]
        obj3.value = newstr