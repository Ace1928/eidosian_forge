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
def bundle_templates(verbose=False, external=True):
    for name, template in param.concrete_descendents(BasicTemplate).items():
        if verbose:
            print(f'Bundling {name} resources')
        if template._resources.get('bundle', True) and external:
            write_component_resources(name, template)
        template_dir = pathlib.Path(inspect.getfile(template)).parent
        dest_dir = BUNDLE_DIR / name.lower()
        dest_dir.mkdir(parents=True, exist_ok=True)
        for css in glob.glob(str(template_dir / '*.css')):
            shutil.copyfile(css, dest_dir / os.path.basename(css))
        template_css = template._css
        if not isinstance(template_css, list):
            template_css = [template_css] if template_css else []
        for css in template_css:
            tmpl_name = name.lower()
            for cls in template.__mro__[1:]:
                if not issubclass(cls, BasicTemplate):
                    continue
                tmpl_css = cls._css if isinstance(cls._css, list) else [cls._css]
                if css in tmpl_css:
                    tmpl_name = cls.__name__.lower()
            tmpl_dest_dir = BUNDLE_DIR / tmpl_name
            tmpl_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(css, tmpl_dest_dir / os.path.basename(css))
        template_js = template._js
        if not isinstance(template_js, list):
            template_js = [template_js] if template_js else []
        for js in template_js:
            tmpl_name = name.lower()
            for cls in template.__mro__[1:]:
                if not issubclass(cls, BasicTemplate):
                    continue
                tmpl_js = cls._js if isinstance(cls._js, list) else [cls._js]
                if js in tmpl_js:
                    tmpl_name = cls.__name__.lower()
            tmpl_dest_dir = BUNDLE_DIR / tmpl_name
            shutil.copyfile(js, tmpl_dest_dir / os.path.basename(js))