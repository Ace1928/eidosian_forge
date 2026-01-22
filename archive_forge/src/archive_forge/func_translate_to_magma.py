import os
import sys
import time
import subprocess
from tempfile import NamedTemporaryFile
import low_index
from low_index import permutation_reps
from low_index import benchmark_util
def translate_to_magma(ex, output):
    output.write('"%s; index = %d";\n' % (ex['group'], ex['index']))
    all_relators = ex['short relators'] + ex['long relators']
    relators = [benchmark_util.expand_relator(r) for r in all_relators]
    letters = 'abcdefghijklmnopqrstuvwxyz'
    generators = letters[:ex['rank']]
    output.write('G := Group<\n')
    output.write(', '.join(['%s' % g for g in generators]))
    output.write(' | ')
    output.write(', '.join(['%s' % r for r in relators]))
    output.write('>;\n')
    output.write('T := Time();\n')
    output.write('sgps := LowIndexSubgroups(G, <1, %d>);\n' % ex['index'])
    output.write('T := Time(T);\n')
    output.write('count := #sgps;\n')
    output.write('printf "%o subgroups\\n", count;\n')
    output.write('printf "%o secs\\n", T;\n')