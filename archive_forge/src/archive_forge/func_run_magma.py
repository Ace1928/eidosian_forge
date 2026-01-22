from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def run_magma(content, filename_base, memory_limit, directory, verbose):
    """
    call magma on the given content and
    """
    magma = magma_executable()
    if magma is None:
        raise ValueError('Sorry, could not find the Magma executable')
    if directory:
        resolved_dir = directory
        if not resolved_dir[-1] == '/':
            resolved_dir = resolved_dir + '/'
    else:
        resolved_dir = tempfile.mkdtemp() + '/'
    in_file = resolved_dir + filename_base + '.magma'
    out_file = resolved_dir + filename_base + '.magma_out'
    if verbose:
        print('Writing to file:', in_file)
    open(in_file, 'wb').write(content.encode('ascii'))
    if verbose:
        print("Magma's output in:", out_file)
    if sys.platform.startswith('win'):
        cmd = 'echo | %s "%s" > "%s"' % (magma, in_file, out_file)
    else:
        cmd = 'ulimit -m %d; echo | %s "%s" > "%s"' % (int(memory_limit / 1024), magma, in_file, out_file)
    if verbose:
        print('Command:', cmd)
        print('Starting magma...')
    retcode = subprocess.call(cmd, shell=True)
    result = open(out_file, 'r').read()
    if verbose:
        print('magma finished.')
        print('Parsing magma result...')
    return decomposition_from_magma(result)