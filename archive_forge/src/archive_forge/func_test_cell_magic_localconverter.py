import pytest
import textwrap
import types
import warnings
from itertools import product
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib._rinterface_capi
import rpy2.robjects
import rpy2.robjects.conversion
from .. import utils
from io import StringIO
from rpy2 import rinterface
from rpy2.robjects import r, vectors, globalenv
import rpy2.robjects.packages as rpacks
@pytest.mark.skipif(IPython is None, reason='The optional package IPython cannot be imported.')
def test_cell_magic_localconverter(ipython_with_magic, clean_globalenv):
    x = (1, 2, 3)
    from rpy2.rinterface import IntSexpVector

    def tuple_str(tpl):
        res = IntSexpVector(tpl)
        return res
    from rpy2.robjects.conversion import Converter
    my_converter = Converter('my converter')
    my_converter.py2rpy.register(tuple, tuple_str)
    from rpy2.robjects import default_converter
    foo = default_converter + my_converter
    snippet = textwrap.dedent('\n    x\n    ')
    ipython_with_magic.push({'x': x})
    with pytest.raises(NameError):
        ipython_with_magic.run_cell_magic('R', '-i x -c foo', snippet)
    ipython_with_magic.push({'x': x, 'foo': 123})
    with pytest.raises(TypeError):
        ipython_with_magic.run_cell_magic('R', '-i x -c foo', snippet)
    ipython_with_magic.push({'x': x, 'foo': foo})
    with pytest.raises(NotImplementedError):
        ipython_with_magic.run_cell_magic('R', '-i x', snippet)
    ipython_with_magic.run_cell_magic('R', '-i x -c foo', snippet)
    ns = types.SimpleNamespace()
    ipython_with_magic.push({'x': x, 'ns': ns})
    with pytest.raises(AttributeError):
        ipython_with_magic.run_cell_magic('R', '-i x -c ns.bar', snippet)
    ns.bar = foo
    ipython_with_magic.run_cell_magic('R', '-i x -c ns.bar', snippet)
    assert isinstance(globalenv['x'], vectors.IntVector)