import os
import textwrap
from passlib.utils.compat import irange
def write_encipher_function(write, indent=0):
    write(indent, '        def encipher(self, l, r):\n            """blowfish encipher a single 64-bit block encoded as two 32-bit ints"""\n\n            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,\n              p10, p11, p12, p13, p14, p15, p16, p17) = self.P\n            S0, S1, S2, S3 = self.S\n\n            l ^= p0\n\n            ')
    render_encipher(write, indent + 1)
    write(indent + 1, '\n        return r ^ p17, l\n\n        ')