import sys
import io
from ase.io import read
from ase.cli.main import CLIError
def header_fmt(c):
    return 'sys2-sys1 image # {}'.format(c)