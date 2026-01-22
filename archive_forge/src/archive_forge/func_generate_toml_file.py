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
def generate_toml_file(project_shortname, pkg_data):
    package_author = pkg_data.get('author', '')
    project_ver = pkg_data.get('version')
    package_name = jl_package_name(project_shortname)
    u = uuid.UUID(jl_dash_uuid)
    package_uuid = uuid.UUID(hex=u.hex[:-12] + hashlib.md5(package_name.encode('utf-8')).hexdigest()[-12:])
    authors_string = 'authors = ["{}"]\n'.format(package_author) if package_author else ''
    base_package = base_package_name(project_shortname)
    toml_string = jl_projecttoml_string.format(package_name=package_name, package_uuid=package_uuid, version=project_ver, authors=authors_string, base_package=base_package, base_version=jl_base_version[base_package], dash_uuid=base_package_uid(project_shortname))
    file_path = 'Project.toml'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(toml_string)
    print('Generated {}'.format(file_path))