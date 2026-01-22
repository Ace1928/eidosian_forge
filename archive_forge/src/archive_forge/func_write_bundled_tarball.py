import fnmatch
import glob
import inspect
import io
import os
import pathlib
import shutil
import tarfile
import zipfile
import param
import requests
from bokeh.model import Model
from .config import config, panel_extension
from .io.resources import RESOURCE_URLS
from .reactive import ReactiveHTML
from .template.base import BasicTemplate
from .theme import Design
def write_bundled_tarball(tarball, name=None, module=False):
    model_name = name.split('.')[-1].lower() if name else ''
    try:
        response = requests.get(tarball['tar'])
    except Exception:
        response = requests.get(tarball['tar'], verify=False)
    f = io.BytesIO()
    f.write(response.content)
    f.seek(0)
    tar_obj = tarfile.open(fileobj=f)
    exclude = tarball.get('exclude', [])
    for tarf in tar_obj:
        if not tarf.name.startswith(tarball['src']) or not tarf.isfile():
            continue
        path = tarf.name.replace(tarball['src'], '')
        if any((fnmatch.fnmatch(tarf.name, exc) for exc in exclude)):
            continue
        bundle_path = os.path.join(*path.split('/'))
        dest_path = tarball['dest'].replace('/', os.path.sep)
        if model_name:
            filename = BUNDLE_DIR.joinpath(model_name, dest_path, bundle_path)
        else:
            filename = BUNDLE_DIR.joinpath(dest_path, bundle_path)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fobj = tar_obj.extractfile(tarf.name)
        filename = str(filename)
        if module and filename.endswith('.js'):
            filename = filename[:-3]
            if filename.endswith('index'):
                filename += '.mjs'
        if any((filename.endswith(ft) for ft in ('.ttf', '.eot', '.woff', '.woff2'))):
            content = fobj.read()
            with open(filename, 'wb') as f:
                f.write(content)
        else:
            content = fobj.read().decode('utf-8')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
    tar_obj.close()