import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def ld_pop_func(self):
    current_pop = None
    line = self.stream.readline().rstrip()
    if line == '':
        self.done = True
        raise StopIteration
    toks = [x for x in line.split(' ') if x != '']
    pop, locus1, locus2 = (toks[0], toks[1], toks[2])
    if not hasattr(self, 'start_locus1'):
        start_locus1, start_locus2 = (locus1, locus2)
        current_pop = -1
    if locus1 == start_locus1 and locus2 == start_locus2:
        current_pop += 1
    if toks[3] == 'No':
        return (current_pop, pop, (locus1, locus2), None)
    p, se, switches = (_gp_float(toks[3]), _gp_float(toks[4]), _gp_int(toks[5]))
    return (current_pop, pop, (locus1, locus2), (p, se, switches))