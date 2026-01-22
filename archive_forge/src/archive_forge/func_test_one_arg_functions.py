import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
def test_one_arg_functions():
    for obj in [g, h, squared]:
        pyfile = dumpIO_source(obj, alias='_obj')
        _obj = loadIO_source(pyfile)
        assert _obj(4) == obj(4)