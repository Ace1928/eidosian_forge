from __future__ import annotations
import json
import os
import os.path as osp
import shutil
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Callable
import pytest
from jupyter_server.serverapp import ServerApp
from jupyterlab_server import LabServerApp
@pytest.fixture
def make_labserver_extension_app(jp_root_dir: Path, jp_template_dir: Path, app_settings_dir: Path, user_settings_dir: Path, schemas_dir: Path, workspaces_dir: Path, labextensions_dir: Path) -> Callable[..., LabServerApp]:
    """Return a factory function for a labserver extension app."""

    def _make_labserver_extension_app(**kwargs: Any) -> LabServerApp:
        """Factory function for lab server extension apps."""
        return LabServerApp(static_dir=str(jp_root_dir), templates_dir=str(jp_template_dir), app_url='/lab', app_settings_dir=str(app_settings_dir), user_settings_dir=str(user_settings_dir), schemas_dir=str(schemas_dir), workspaces_dir=str(workspaces_dir), extra_labextensions_path=[str(labextensions_dir)])
    index = jp_template_dir.joinpath('index.html')
    index.write_text('\n<!DOCTYPE html>\n<html>\n<head>\n  <title>{{page_config[\'appName\'] | e}}</title>\n</head>\n<body>\n    {# Copy so we do not modify the page_config with updates. #}\n    {% set page_config_full = page_config.copy() %}\n\n    {# Set a dummy variable - we just want the side effect of the update. #}\n    {% set _ = page_config_full.update(baseUrl=base_url, wsUrl=ws_url) %}\n\n      <script id="jupyter-config-data" type="application/json">\n        {{ page_config_full | tojson }}\n      </script>\n  <script src="{{page_config[\'fullStaticUrl\'] | e}}/bundle.js" main="index"></script>\n\n  <script type="text/javascript">\n    /* Remove token from URL. */\n    (function () {\n      var parsedUrl = new URL(window.location.href);\n      if (parsedUrl.searchParams.get(\'token\')) {\n        parsedUrl.searchParams.delete(\'token\');\n        window.history.replaceState({ }, \'\', parsedUrl.href);\n      }\n    })();\n  </script>\n</body>\n</html>\n')
    src = pjoin(HERE, 'test_data', 'schemas', '@jupyterlab')
    dst = pjoin(str(schemas_dir), '@jupyterlab')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for name in ['apputils-extension', 'codemirror-extension']:
        target_name = name + '-federated'
        target = pjoin(str(labextensions_dir), '@jupyterlab', target_name)
        src = pjoin(HERE, 'test_data', 'schemas', '@jupyterlab', name)
        dst = pjoin(target, 'schemas', '@jupyterlab', target_name)
        if osp.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        with open(pjoin(target, 'package.orig.json'), 'w') as fid:
            data = dict(name=target_name, jupyterlab=dict(extension=True))
            json.dump(data, fid)
    src = pjoin(HERE, 'test_data', 'app-settings', 'overrides.json')
    dst = pjoin(str(app_settings_dir), 'overrides.json')
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)
    ws_path = pjoin(HERE, 'test_data', 'workspaces')
    for item in os.listdir(ws_path):
        src = pjoin(ws_path, item)
        dst = pjoin(str(workspaces_dir), item)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy(src, str(workspaces_dir))
    return _make_labserver_extension_app