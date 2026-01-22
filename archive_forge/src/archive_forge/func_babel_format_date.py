import os
import re
import warnings
from datetime import datetime, timezone
from os import path
from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
import babel.dates
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import SEP, canon_path, relpath
def babel_format_date(date: datetime, format: str, locale: str, formatter: Callable=babel.dates.format_date) -> str:
    if locale is None:
        warnings.warn('The locale argument for babel_format_date() becomes required.', RemovedInSphinx70Warning)
        locale = 'en'
    if not hasattr(date, 'tzinfo'):
        formatter = babel.dates.format_date
    try:
        return formatter(date, format, locale=locale)
    except (ValueError, babel.core.UnknownLocaleError):
        return formatter(date, format, locale='en')
    except AttributeError:
        logger.warning(__('Invalid date format. Quote the string by single quote if you want to output it directly: %s'), format)
        return format