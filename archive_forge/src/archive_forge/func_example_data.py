import os
from . import decorators
from .utils import package_check, TempFATFS
def example_data(infile='functional.nii'):
    """returns path to empty example data files for doc tests
    it will raise an exception if filename is not in the directory"""
    filepath = os.path.abspath(__file__)
    basedir = os.path.dirname(filepath)
    outfile = os.path.join(basedir, 'data', infile)
    if not os.path.exists(outfile):
        raise IOError('%s empty data file does NOT exist' % outfile)
    return outfile