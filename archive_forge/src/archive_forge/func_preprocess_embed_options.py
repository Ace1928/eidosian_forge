from typing import Literal, Optional, Union, cast, Tuple
from .deprecation import AltairDeprecationWarning
from .html import spec_to_html
from ._importers import import_vl_convert, vl_version_for_vl_convert
import struct
import warnings
def preprocess_embed_options(embed_options: dict) -> dict:
    """Preprocess embed options to a form compatible with Vega Embed

    Parameters
    ----------
    embed_options : dict
        The embed options dictionary to preprocess.

    Returns
    -------
    embed_opts : dict
        The preprocessed embed options dictionary.
    """
    embed_options = (embed_options or {}).copy()
    format_locale = embed_options.get('formatLocale', None)
    if isinstance(format_locale, str):
        vlc = import_vl_convert()
        embed_options['formatLocale'] = vlc.get_format_locale(format_locale)
    time_format_locale = embed_options.get('timeFormatLocale', None)
    if isinstance(time_format_locale, str):
        vlc = import_vl_convert()
        embed_options['timeFormatLocale'] = vlc.get_time_format_locale(time_format_locale)
    return embed_options