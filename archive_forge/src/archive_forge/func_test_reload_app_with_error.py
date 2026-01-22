import os
import pathlib
import time
import pytest
from panel.io.state import state
from panel.tests.util import serve_component
def test_reload_app_with_error(page, autoreload, py_file):
    py_file.write("import panel as pn; pn.panel('foo').servable();")
    py_file.close()
    path = pathlib.Path(py_file.name)
    autoreload(path)
    serve_component(page, path)
    expect(page.locator('.markdown')).to_have_text('foo')
    with open(py_file.name, 'w') as f:
        f.write('foo+bar')
        os.fsync(f)
    expect(page.locator('.alert')).to_have_count(1)