from io import StringIO
import pytest
from panel.io.handlers import capture_code_cell, extract_code, parse_notebook
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import panel"""
def test_capture_code_cell_expression_with_comment():
    assert capture_code_cell({'id': 'foo', 'source': code_expr_comment}) == ['', "_pn__state._cell_outputs['foo'].append((1+1))\nfor _cell__out in _CELL__DISPLAY:\n    _pn__state._cell_outputs['foo'].append(_cell__out)\n_CELL__DISPLAY.clear()\n_fig__out = _get__figure()\nif _fig__out:\n    _pn__state._cell_outputs['foo'].append(_fig__out)\n"]