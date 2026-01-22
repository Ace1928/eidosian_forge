from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_astore(self):
    array = self.pop('arraytype')
    for i in range(len(array.value) - 1, -1, -1):
        array.value[i] = self.pop()
    self.push(array)