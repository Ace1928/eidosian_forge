import os
import warnings
from importlib import import_module
from pathlib import Path
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.conf import closest_scrapy_cfg, get_config, init_env
def data_path(path: str, createdir: bool=False) -> str:
    """
    Return the given path joined with the .scrapy data directory.
    If given an absolute path, return it unmodified.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        if inside_project():
            path_obj = Path(project_data_dir(), path)
        else:
            path_obj = Path('.scrapy', path)
    if createdir and (not path_obj.exists()):
        path_obj.mkdir(parents=True)
    return str(path_obj)