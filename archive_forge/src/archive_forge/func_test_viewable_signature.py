import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
@pytest.mark.parametrize('viewable', all_viewables)
def test_viewable_signature(viewable):
    from inspect import Parameter, signature
    parameters = signature(viewable).parameters
    if getattr(getattr(viewable, '_param__private', object), 'signature', None):
        pytest.skip('Signature already set by Param')
    assert 'params' in parameters
    try:
        assert parameters['params'] == Parameter('params', Parameter.VAR_KEYWORD, annotation='Any')
    except Exception:
        assert parameters['params'] == Parameter('params', Parameter.VAR_KEYWORD)