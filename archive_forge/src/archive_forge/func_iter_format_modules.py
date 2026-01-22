import datetime
import decimal
import functools
import re
import unicodedata
from importlib import import_module
from django.conf import settings
from django.utils import dateformat, numberformat
from django.utils.functional import lazy
from django.utils.translation import check_for_language, get_language, to_locale
def iter_format_modules(lang, format_module_path=None):
    """Find format modules."""
    if not check_for_language(lang):
        return
    if format_module_path is None:
        format_module_path = settings.FORMAT_MODULE_PATH
    format_locations = []
    if format_module_path:
        if isinstance(format_module_path, str):
            format_module_path = [format_module_path]
        for path in format_module_path:
            format_locations.append(path + '.%s')
    format_locations.append('django.conf.locale.%s')
    locale = to_locale(lang)
    locales = [locale]
    if '_' in locale:
        locales.append(locale.split('_')[0])
    for location in format_locations:
        for loc in locales:
            try:
                yield import_module('%s.formats' % (location % loc))
            except ImportError:
                pass