from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_getinterval(self):
    obj1 = self.pop('integertype')
    obj2 = self.pop('integertype')
    obj3 = self.pop('arraytype', 'stringtype')
    tp = obj3.type
    if tp == 'arraytype':
        self.push(ps_array(obj3.value[obj2.value:obj2.value + obj1.value]))
    elif tp == 'stringtype':
        self.push(ps_string(obj3.value[obj2.value:obj2.value + obj1.value]))