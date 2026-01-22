import os
import json
from jupyter_core.paths import jupyter_path
import nbconvert.exporters.templateexporter
def collect_paths(app_names, template_name='default', subdir=None, include_root_paths=True, prune=False, root_dirs=None):
    """
    Voilà supports custom templates for rendering notebooks.
    For a specified template name, `collect_paths` can be used to collects
        - template paths
        - resources paths (by using the subdir arg)

    by looking in the standard Jupyter data directories:
    $PREFIX/share/jupyter/templates/<app_name>/<template_name>[/subdir]
    with different prefix values (user directory, sys prefix, and then system prefix) which
    allows users to override templates locally.
    The function will recursively load the base templates upon which the specified template
    may be based.
    """
    found_at_least_one = False
    paths = []
    full_paths = []
    root_dirs = root_dirs or _default_root_dirs()
    template_names = _find_template_hierarchy(app_names, template_name, root_dirs)
    for template_name in template_names:
        for root_dir in root_dirs:
            for app_name in app_names:
                app_dir = os.path.join(root_dir, app_name, 'templates')
                path = os.path.join(app_dir, template_name)
                if subdir:
                    path = os.path.join(path, subdir)
                if not prune or os.path.exists(path):
                    paths.append(path)
                    found_at_least_one = True
    if include_root_paths:
        for root_dir in root_dirs:
            paths.append(root_dir)
            for app_name in app_names:
                app_dir = os.path.join(root_dir, app_name, 'templates')
                paths.append(app_dir)
    if not found_at_least_one:
        paths = '\n\t'.join(full_paths)
        raise ValueError('No template sub-directory with name %r found in the following paths:\n\t%s' % (template_name, paths))
    return paths