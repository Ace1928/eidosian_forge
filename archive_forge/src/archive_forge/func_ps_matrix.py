from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_matrix(self):
    matrix = [ps_real(1.0), ps_integer(0), ps_integer(0), ps_real(1.0), ps_integer(0), ps_integer(0)]
    self.push(ps_array(matrix))