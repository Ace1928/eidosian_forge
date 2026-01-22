import re
import unicodedata
from typing import Iterator
from emoji import unicode_codes
from emoji.tokenizer import Token, EmojiMatch, EmojiMatchZWJ, EmojiMatchZWJNonRGI, tokenize, filter_tokens
def emojize(string, delimiters=(_DEFAULT_DELIMITER, _DEFAULT_DELIMITER), variant=None, language='en', version=None, handle_version=None):
    """
    Replace emoji names in a string with Unicode codes.
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbsup:", language='alias'))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun {thumbs_up}", delimiters = ("{", "}")))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun :red_heart:", variant="text_type"))
        Python is fun ❤
        >>> print(emoji.emojize("Python is fun :red_heart:", variant="emoji_type"))
        Python is fun ❤️ # red heart, not black heart

    :param string: String contains emoji names.
    :param delimiters: (optional) Use delimiters other than _DEFAULT_DELIMITER. Each delimiter
        should contain at least one character that is not part of a-zA-Z0-9 and ``_-&.()!?#*+,``.
        See ``emoji.core._EMOJI_NAME_PATTERN`` for the regular expression of unsafe characters.
    :param variant: (optional) Choose variation selector between "base"(None), VS-15 ("text_type") and VS-16 ("emoji_type")
    :param language: Choose language of emoji name: language code 'es', 'de', etc. or 'alias'
        to use English aliases
    :param version: (optional) Max version. If set to an Emoji Version,
        all emoji above this version will be ignored.
    :param handle_version: (optional) Replace the emoji above ``version``
        instead of ignoring it. handle_version can be either a string or a
        callable; If it is a callable, it's passed the Unicode emoji and the
        data dict from :data:`EMOJI_DATA` and must return a replacement string
        to be used::

            handle_version('\\U0001F6EB', {
                'en' : ':airplane_departure:',
                'status' : fully_qualified,
                'E' : 1,
                'alias' : [':flight_departure:'],
                'de': ':abflug:',
                'es': ':avión_despegando:',
                ...
            })

    :raises ValueError: if ``variant`` is neither None, 'text_type' or 'emoji_type'

    """
    if language == 'alias':
        language_pack = unicode_codes.get_aliases_unicode_dict()
    else:
        language_pack = unicode_codes.get_emoji_unicode_dict(language)
    pattern = re.compile('(%s[%s]+%s)' % (re.escape(delimiters[0]), _EMOJI_NAME_PATTERN, re.escape(delimiters[1])))

    def replace(match):
        name = match.group(1)[len(delimiters[0]):-len(delimiters[1])]
        emj = language_pack.get(_DEFAULT_DELIMITER + unicodedata.normalize('NFKC', name) + _DEFAULT_DELIMITER)
        if emj is None:
            return match.group(1)
        if version is not None and unicode_codes.EMOJI_DATA[emj]['E'] > version:
            if callable(handle_version):
                emj_data = unicode_codes.EMOJI_DATA[emj].copy()
                emj_data['match_start'] = match.start()
                emj_data['match_end'] = match.end()
                return handle_version(emj, emj_data)
            elif handle_version is not None:
                return str(handle_version)
            else:
                return ''
        if variant is None or 'variant' not in unicode_codes.EMOJI_DATA[emj]:
            return emj
        if emj[-1] == '︎' or emj[-1] == '️':
            emj = emj[0:-1]
        if variant == 'text_type':
            return emj + '︎'
        elif variant == 'emoji_type':
            return emj + '️'
        else:
            raise ValueError("Parameter 'variant' must be either None, 'text_type' or 'emoji_type'")
    return pattern.sub(replace, string)