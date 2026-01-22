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
def write_bundled_zip(name, resource):
    try:
        response = requests.get(resource['zip'])
    except Exception:
        response = requests.get(resource['zip'], verify=False)
    f = io.BytesIO()
    f.write(response.content)
    f.seek(0)
    zip_obj = zipfile.ZipFile(f)
    exclude = resource.get('exclude', [])
    for zipf in zip_obj.namelist():
        path = zipf.replace(resource['src'], '')
        if any((fnmatch.fnmatch(zipf, exc) for exc in exclude)) or zipf.endswith('/'):
            continue
        bundle_path = path.replace('/', os.path.sep)
        filename = BUNDLE_DIR.joinpath(name, bundle_path)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fdata = zip_obj.read(zipf)
        filename = str(filename)
        if any((filename.endswith(ft) for ft in ('.ttf', '.eot', '.woff', '.woff2'))):
            with open(filename, 'wb') as f:
                f.write(fdata)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(fdata.decode('utf-8'))
    zip_obj.close()