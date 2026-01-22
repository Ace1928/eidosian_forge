from io import StringIO
import pytest
from panel.io.handlers import capture_code_cell, extract_code, parse_notebook
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import panel"""
def test_capture_code_cell_expression_semicolon():
    assert capture_code_cell({'id': 'foo', 'source': code_expr_semicolon}) == ['', '1+1;']