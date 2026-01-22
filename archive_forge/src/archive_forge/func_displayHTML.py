import base64
import json
import webbrowser
import inspect
import os
from os.path import isdir
from plotly import utils, optional_imports
from plotly.io import to_json, to_image, write_image, write_html
from plotly.io._orca import ensure_server
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import _get_jconfig, get_plotlyjs
from plotly.tools import return_figure_from_figure_or_data
@property
def displayHTML(self):
    import inspect
    if self._displayHTML is None:
        for frame in inspect.getouterframes(inspect.currentframe()):
            global_names = set(frame.frame.f_globals)
            if all((v in global_names for v in ['displayHTML', 'display', 'spark'])):
                self._displayHTML = frame.frame.f_globals['displayHTML']
                break
        if self._displayHTML is None:
            raise EnvironmentError("\nUnable to detect the Databricks displayHTML function. The 'databricks' renderer is only\nsupported when called from within the Databricks notebook environment.")
    return self._displayHTML