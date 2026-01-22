from __future__ import (absolute_import, division, print_function)
import sys
import warnings
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
def encode_txt_value(value, always_quote=False, use_character_encoding=_SENTINEL, use_octal=_SENTINEL, character_encoding=_SENTINEL):
    """
    Given a decoded TXT value, encodes it.

    If always_quote is set to True, always use double quotes for all strings.
    If use_character_encoding (default: True) is set to False, do not use octal encoding.
    """
    if use_octal is not _SENTINEL:
        warnings.warn('The encode_txt_value parameter use_octal is deprecated. Use use_character_encoding instead.', DeprecationWarning)
        if use_character_encoding is not _SENTINEL:
            raise ValueError('Cannot use both use_character_encoding and use_octal. Use only use_character_encoding!')
        use_character_encoding = use_octal
    if use_character_encoding is _SENTINEL:
        use_character_encoding = True
    if character_encoding is _SENTINEL:
        warnings.warn('The default value of the encode_txt_value parameter character_encoding is deprecated. Set explicitly to "octal" for the old behavior, or set to "decimal" for the new and correct behavior.', DeprecationWarning)
        character_encoding = 'octal'
    if character_encoding not in ('octal', 'decimal'):
        raise ValueError('character_encoding must be set to "octal" or "decimal"')
    value = to_bytes(value)
    buffer = []
    output = []

    def append(buffer):
        value = b''.join(buffer)
        if b' ' in value or not value or always_quote:
            value = b'"%s"' % value
        output.append(value)
    index = 0
    length = len(value)
    while index < length:
        letter = value[index:index + 1]
        index += 1
        if letter in (b'"', b'\\'):
            if len(buffer) + 2 > 255:
                append(buffer[:255])
                buffer = buffer[255:]
            buffer.append(b'\\')
            buffer.append(letter)
        elif use_character_encoding and (not 32 <= ord(letter) < 127):
            if len(buffer) + 4 > 255:
                append(buffer[:255])
                buffer = buffer[255:]
            letter_value = ord(letter)
            buffer.append(b'\\')
            if character_encoding == 'octal':
                v2 = letter_value >> 6 & 7
                v1 = letter_value >> 3 & 7
                v0 = letter_value & 7
            else:
                v2 = letter_value // 100 % 10
                v1 = letter_value // 10 % 10
                v0 = letter_value % 10
            buffer.append(_DECIMAL_DIGITS[v2:v2 + 1])
            buffer.append(_DECIMAL_DIGITS[v1:v1 + 1])
            buffer.append(_DECIMAL_DIGITS[v0:v0 + 1])
        elif not use_character_encoding and ord(letter) & 128 != 0:
            utf8_length = min(_get_utf8_length(ord(letter)), length - index + 1)
            if len(buffer) + utf8_length > 255:
                append(buffer[:255])
                buffer = buffer[255:]
            buffer.append(letter)
            while utf8_length > 1:
                buffer.append(value[index:index + 1])
                index += 1
                utf8_length -= 1
        else:
            buffer.append(letter)
        if len(buffer) >= 255:
            append(buffer[:255])
            buffer = buffer[255:]
    if buffer or not output:
        append(buffer)
    return to_text(b' '.join(output))