import os
import gettext
from jinja2 import Environment, FileSystemLoader
from jupyter_server.utils import url_path_join
from jupyter_server.base.handlers import path_regex, FileFindHandler
from jupyterlab_server.themes_handler import ThemesHandler
from .paths import ROOT, collect_template_paths, collect_static_paths, jupyter_path
from .handler import VoilaHandler
from .treehandler import VoilaTreeHandler
from .static_file_handler import MultiStaticFileHandler, TemplateStaticFileHandler, WhiteListFileHandler
from .configuration import VoilaConfiguration
from .utils import get_server_root_dir
from .shutdown_kernel_handler import VoilaShutdownKernelHandler

    Returns a list of dictionaries with metadata describing
    where to find the `_load_jupyter_server_extension` function.
    