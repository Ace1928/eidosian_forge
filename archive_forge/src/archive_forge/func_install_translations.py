import gettext as _gettext
import os
import sys
def install_translations(lang=None, domain='brz', locale_base=None):
    """Create a gettext translation object.

    :param lang: language to install.
    :param domain: translation domain to install.
    :param locale_base: plugins can specify their own directory.

    :returns: a gettext translations object to use
    """
    if lang is None:
        lang = _get_current_locale()
    if lang is not None:
        languages = lang.split(':')
    else:
        languages = None
    translation = _gettext.translation(domain, localedir=_get_locale_dir(locale_base), languages=languages, fallback=True)
    return translation