from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_eq(self):
    any1 = self.pop()
    any2 = self.pop()
    self.push(ps_boolean(any1.value == any2.value))