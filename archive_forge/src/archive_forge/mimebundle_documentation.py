from typing import Literal, Optional, Union, cast, Tuple
from .deprecation import AltairDeprecationWarning
from .html import spec_to_html
from ._importers import import_vl_convert, vl_version_for_vl_convert
import struct
import warnings
Preprocess embed options to a form compatible with Vega Embed

    Parameters
    ----------
    embed_options : dict
        The embed options dictionary to preprocess.

    Returns
    -------
    embed_opts : dict
        The preprocessed embed options dictionary.
    