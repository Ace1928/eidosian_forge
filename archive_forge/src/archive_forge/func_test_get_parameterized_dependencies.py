import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_get_parameterized_dependencies():

    class Test(param.Parameterized):
        a = param.Parameter()
        b = param.Parameter()

        @param.depends('a')
        def dep_a(self):
            return

        @param.depends('dep_a', 'b')
        def dep_ab(self):
            return
    test = Test()
    assert extract_dependencies(test.dep_a) == [test.param.a]
    assert extract_dependencies(test.dep_ab) == [test.param.a, test.param.b]