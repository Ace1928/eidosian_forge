import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def generate_init_offline_itables_html(dt_bundle: Path):
    assert dt_bundle.suffix == '.js'
    dt_src = dt_bundle.read_text()
    dt_css = dt_bundle.with_suffix('.css').read_text()
    dt64 = b64encode(dt_src.encode('utf-8')).decode('ascii')
    return f'<style>{dt_css}</style>\n<script>window.{DATATABLES_SRC_FOR_ITABLES} = "data:text/javascript;base64,{dt64}"</script>\n'