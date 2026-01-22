def replace_s(self, c_bra, c_ket, s):
    """
        to replace chars between c_bra and c_ket in self.current by the
        chars in s.

        @type c_bra int
        @type c_ket int
        @type s: string
        """
    adjustment = len(s) - (c_ket - c_bra)
    self.current = self.current[0:c_bra] + s + self.current[c_ket:]
    self.limit += adjustment
    if self.cursor >= c_ket:
        self.cursor += adjustment
    elif self.cursor > c_bra:
        self.cursor = c_bra
    return adjustment