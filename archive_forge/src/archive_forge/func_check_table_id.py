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
def check_table_id(table_id):
    """Make sure that the table_id is a valid HTML id.

    See also https://stackoverflow.com/questions/70579/html-valid-id-attribute-values
    """
    if table_id is not None:
        if not re.match('[A-Za-z][-A-Za-z0-9_.]*', table_id):
            raise ValueError(f'The id name must contain at least one character, cannot start with a number, and must not contain whitespaces ({table_id})')