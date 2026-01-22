import os
import json
from jupyter_core.paths import jupyter_path
import nbconvert.exporters.templateexporter
def collect_template_paths(app_names, template_name='default', prune=False, root_dirs=None):
    return collect_paths(app_names, template_name, include_root_paths=True, prune=prune, root_dirs=root_dirs)