import os
import sys
import tokenize
def errprint(*args):
    sep = ''
    for arg in args:
        sys.stderr.write(sep + str(arg))
        sep = ' '
    sys.stderr.write('\n')