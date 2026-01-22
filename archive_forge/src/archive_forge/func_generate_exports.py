import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def generate_exports(project_shortname, components, metadata, pkg_data, rpkg_data, prefix, package_depends, package_imports, package_suggests, **kwargs):
    export_string = make_namespace_exports(components, prefix)
    has_wildcards = False
    for component_data in metadata.values():
        if any((key.endswith('-*') for key in component_data['props'])):
            has_wildcards = True
            break
    generate_rpkg(pkg_data, rpkg_data, project_shortname, export_string, package_depends, package_imports, package_suggests, has_wildcards)