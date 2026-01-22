import re
import unicodedata
from typing import Iterator
from emoji import unicode_codes
from emoji.tokenizer import Token, EmojiMatch, EmojiMatchZWJ, EmojiMatchZWJNonRGI, tokenize, filter_tokens
def demojize(string, delimiters=(_DEFAULT_DELIMITER, _DEFAULT_DELIMITER), language='en', version=None, handle_version=None):
    """
    Replace Unicode emoji in a string with emoji shortcodes. Useful for storage.
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.demojize("Python is fun 👍"))
        Python is fun :thumbs_up:
        >>> print(emoji.demojize("icode is tricky 😯", delimiters=("__", "__")))
        Unicode is tricky __hushed_face__

    :param string: String contains Unicode characters. MUST BE UNICODE.
    :param delimiters: (optional) User delimiters other than ``_DEFAULT_DELIMITER``
    :param language: Choose language of emoji name: language code 'es', 'de', etc. or 'alias'
        to use English aliases
    :param version: (optional) Max version. If set to an Emoji Version,
        all emoji above this version will be removed.
    :param handle_version: (optional) Replace the emoji above ``version``
        instead of removing it. handle_version can be either a string or a
        callable ``handle_version(emj: str, data: dict) -> str``; If it is
        a callable, it's passed the Unicode emoji and the data dict from
        :data:`EMOJI_DATA` and must return a replacement string  to be used.
        The passed data is in the form of::

            handle_version('\\U0001F6EB', {
                'en' : ':airplane_departure:',
                'status' : fully_qualified,
                'E' : 1,
                'alias' : [':flight_departure:'],
                'de': ':abflug:',
                'es': ':avión_despegando:',
                ...
            })

    """
    if language == 'alias':
        language = 'en'
        _use_aliases = True
    else:
        _use_aliases = False

    def handle(emoji_match):
        if version is not None and emoji_match.data['E'] > version:
            if callable(handle_version):
                return handle_version(emoji_match.emoji, emoji_match.data_copy())
            elif handle_version is not None:
                return handle_version
            else:
                return ''
        elif language in emoji_match.data:
            if _use_aliases and 'alias' in emoji_match.data:
                return delimiters[0] + emoji_match.data['alias'][0][1:-1] + delimiters[1]
            else:
                return delimiters[0] + emoji_match.data[language][1:-1] + delimiters[1]
        else:
            return emoji_match.emoji
    matches = tokenize(string, keep_zwj=config.demojize_keep_zwj)
    return ''.join((str(handle(token.value)) if isinstance(token.value, EmojiMatch) else token.value for token in matches))