import gettext
import logging
import os
import sqlite3
import sys
def code_to_name(code, separator='_'):
    """
    Get the human readable and translated name of a language based on it's code.

    :param code: the code of the language (e.g. de_DE, en_US)
    :param target: separator used to separate language from country
    :rtype: human readable and translated language name
    """
    logger.debug('requesting name for code "{}"'.format(code))
    code = code.split(separator)
    if len(code) > 1:
        lang = Language.by_iso_639_1(code[0]).translation
        country = Country.by_alpha_2(code[1]).translation
        return '{} ({})'.format(lang, country)
    else:
        return Language.by_iso_639_1(code[0]).translation