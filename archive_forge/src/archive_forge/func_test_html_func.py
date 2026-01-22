import pytest
import types
import warnings
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr
@pytest.mark.skipif(IPython is None, reason='The optional package IPython cannot be imported.')
@pytest.mark.parametrize('o,func', [(vectors.IntVector([1, 2, 3]), html.html_vector_horizontal), (vectors.FloatVector([1, 2, 3]), html.html_vector_horizontal), (vectors.StrVector(['a', 'bc']), html.html_vector_horizontal), (vectors.FactorVector(['a', 'bc']), html.html_vector_horizontal), (vectors.ListVector({'a': 1, 'b': 2}), html.html_rlist), (vectors.DataFrame({'a': 1, 'b': 'z'}), html.html_rdataframe), ('x <- c(1, 2, 3)', html.html_sourcecode), (base.c, html.html_ridentifiedobject)])
def test_html_func(o, func):
    res = func(o)
    assert isinstance(res, str)