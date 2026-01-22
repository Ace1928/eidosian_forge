import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_get_parameterized_subobject_dependencies():

    class A(param.Parameterized):
        value = param.Parameter()

    class B(param.Parameterized):
        a = param.ClassSelector(default=A(), class_=A)

        @param.depends('a.value')
        def dep_a_value(self):
            return
    test = B()
    assert extract_dependencies(test.dep_a_value) == [test.a.param.value]