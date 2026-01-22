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
def generate_package_file(project_shortname, components, pkg_data, prefix):
    package_name = jl_package_name(project_shortname)
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(project_shortname)
    js_dist = getattr(mod, '_js_dist', [])
    css_dist = getattr(mod, '_css_dist', [])
    project_ver = pkg_data.get('version')
    resources_dist = ',\n'.join(generate_metadata_strings(js_dist, 'js') + generate_metadata_strings(css_dist, 'css'))
    package_string = jl_package_file_string.format(package_name=package_name, component_includes='\n'.join([jl_component_include_string.format(name=format_fn_name(prefix, comp_name)) for comp_name in components]), resources_dist=resources_dist, version=project_ver, project_shortname=project_shortname, base_package=base_package_name(project_shortname))
    file_path = os.path.join('src', package_name + '.jl')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(package_string)
    print('Generated {}'.format(file_path))