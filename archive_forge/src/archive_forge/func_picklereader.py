from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.compat import pickle
from petl.test.helpers import ieq
from petl.io.pickle import frompickle, topickle, appendpickle
def picklereader(fl):
    try:
        while True:
            yield pickle.load(fl)
    except EOFError:
        pass