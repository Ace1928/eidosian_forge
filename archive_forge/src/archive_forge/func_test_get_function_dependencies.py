import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_get_function_dependencies():

    class Test(param.Parameterized):
        a = param.Parameter()
    assert extract_dependencies(bind(lambda a: a, Test.param.a)) == [Test.param.a]