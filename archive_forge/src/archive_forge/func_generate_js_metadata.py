import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def generate_js_metadata(pkg_data, project_shortname):
    """Dynamically generate R function to supply JavaScript and CSS dependency
    information required by the dash package for R.

    Parameters
    ----------
    project_shortname = component library name, in snake case

    Returns
    -------
    function_string = complete R function code to provide component features
    """
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(project_shortname)
    alldist = getattr(mod, '_js_dist', []) + getattr(mod, '_css_dist', [])
    project_ver = pkg_data.get('version')
    rpkgname = snake_case_to_camel_case(project_shortname)
    function_frame_open = frame_open_template.format(rpkgname=rpkgname)
    function_frame = []
    function_frame_body = []
    if len(alldist) > 1:
        for dep in range(len(alldist)):
            curr_dep = alldist[dep]
            rpp = curr_dep['relative_package_path']
            async_or_dynamic = get_async_type(curr_dep)
            if 'dash_' in rpp:
                dep_name = rpp.split('.')[0]
            else:
                dep_name = '{}'.format(project_shortname)
            if 'css' in rpp:
                css_name = "'{}'".format(rpp)
                script_name = 'NULL'
            else:
                script_name = "'{}'".format(rpp)
                css_name = 'NULL'
            function_frame += [frame_element_template.format(dep_name=dep_name, project_ver=project_ver, rpkgname=rpkgname, project_shortname=project_shortname, script_name=script_name, css_name=css_name, async_or_dynamic=async_or_dynamic)]
            function_frame_body = ',\n'.join(function_frame)
    elif len(alldist) == 1:
        dep = alldist[0]
        rpp = dep['relative_package_path']
        async_or_dynamic = get_async_type(dep)
        if 'css' in rpp:
            css_name = "'{}'".format(rpp)
            script_name = 'NULL'
        else:
            script_name = "'{}'".format(rpp)
            css_name = 'NULL'
        function_frame_body = frame_body_template.format(project_shortname=project_shortname, project_ver=project_ver, rpkgname=rpkgname, script_name=script_name, css_name=css_name, async_or_dynamic=async_or_dynamic)
    function_string = ''.join([function_frame_open, function_frame_body, frame_close_template])
    return function_string