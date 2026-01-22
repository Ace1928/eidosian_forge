import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def download_plotlyjs(download_url):
    warnings.warn('\n        `download_plotlyjs` is deprecated and will be removed in the\n        next release. plotly.js is shipped with this module, it is no\n        longer necessary to download this bundle separately.\n    ', DeprecationWarning)