import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def generate_module(project_shortname, components, metadata, pkg_data, prefix, **kwargs):
    if os.path.exists('deps'):
        shutil.rmtree('deps')
    os.makedirs('deps')
    for rel_dirname, _, filenames in os.walk(project_shortname):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension in ['.py', '.pyc', '.json']:
                continue
            target_dirname = os.path.join('deps/', os.path.relpath(rel_dirname, project_shortname))
            if not os.path.exists(target_dirname):
                os.makedirs(target_dirname)
            shutil.copy(os.path.join(rel_dirname, filename), target_dirname)
    generate_package_file(project_shortname, components, pkg_data, prefix)
    generate_toml_file(project_shortname, pkg_data)