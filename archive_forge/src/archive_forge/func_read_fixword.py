def read_fixword(self):
    word = self.read_word()
    neg = False
    if word & 2147483648:
        neg = True
        word = -word & 4294967295
    return (-1 if neg else 1) * word / float(1 << 20)