import os
import tempfile
import pytest
import requests
from panel.tests.util import (
import panel as pn
@linux_only
def test_serve_admin_custom_endpoint(py_file):
    app = "import panel as pn; pn.Row('# Example').servable(title='A')"
    write_file(app, py_file.file)
    with run_panel_serve(['--port', '0', '--admin', '--admin-endpoint', 'foo', py_file.name]) as p:
        port = wait_for_port(p.stdout)
        r = requests.get(f'http://localhost:{port}/foo')
        assert r.status_code == 200
        assert 'Admin' in r.content.decode('utf-8')
        r2 = requests.get(f'http://localhost:{port}/admin')
        assert r2.status_code == 404