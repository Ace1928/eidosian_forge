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
def number_format(value, decimal_pos=None, use_l10n=None, force_grouping=False):
    """
    Format a numeric value using localization settings.

    If use_l10n is provided and is not None, it forces the value to
    be localized (or not), otherwise it's always localized.
    """
    if use_l10n is None:
        use_l10n = True
    lang = get_language() if use_l10n else None
    return numberformat.format(value, get_format('DECIMAL_SEPARATOR', lang, use_l10n=use_l10n), decimal_pos, get_format('NUMBER_GROUPING', lang, use_l10n=use_l10n), get_format('THOUSAND_SEPARATOR', lang, use_l10n=use_l10n), force_grouping=force_grouping, use_l10n=use_l10n)