import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def testdir(a):
    try:
        names = [n for n in os.listdir(a) if n.endswith('.py')]
    except OSError:
        sys.stderr.write('Directory not readable: %s' % a)
    else:
        for n in names:
            fullname = os.path.join(a, n)
            if os.path.isfile(fullname):
                output = io.StringIO()
                print('Testing %s' % fullname)
                try:
                    roundtrip(fullname, output)
                except Exception as e:
                    print('  Failed to compile, exception is %s' % repr(e))
            elif os.path.isdir(fullname):
                testdir(fullname)