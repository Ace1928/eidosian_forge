from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def get_supported_language_variant(lang_code, *, strict=False):
    return _trans.get_supported_language_variant(lang_code, strict)