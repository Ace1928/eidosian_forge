from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def get_language_info(lang_code):
    from django.conf.locale import LANG_INFO
    try:
        lang_info = LANG_INFO[lang_code]
        if 'fallback' in lang_info and 'name' not in lang_info:
            info = get_language_info(lang_info['fallback'][0])
        else:
            info = lang_info
    except KeyError:
        if '-' not in lang_code:
            raise KeyError('Unknown language code %s.' % lang_code)
        generic_lang_code = lang_code.split('-')[0]
        try:
            info = LANG_INFO[generic_lang_code]
        except KeyError:
            raise KeyError('Unknown language code %s and %s.' % (lang_code, generic_lang_code))
    if info:
        info['name_translated'] = gettext_lazy(info['name'])
    return info