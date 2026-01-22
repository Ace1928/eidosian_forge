import re
import sys
import subprocess
def output_def(dlist, flist, header, file=sys.stdout):
    """Outputs the final DEF file to a file defaulting to stdout.

output_def(dlist, flist, header, file = sys.stdout)"""
    for data_sym in dlist:
        header = header + '\t%s DATA\n' % data_sym
    header = header + '\n'
    for func_sym in flist:
        header = header + '\t%s\n' % func_sym
    file.write(header)