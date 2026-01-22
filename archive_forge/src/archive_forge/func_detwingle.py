from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def detwingle(cls, in_bytes, main_encoding='utf8', embedded_encoding='windows-1252'):
    """Fix characters from one encoding embedded in some other encoding.

        Currently the only situation supported is Windows-1252 (or its
        subset ISO-8859-1), embedded in UTF-8.

        :param in_bytes: A bytestring that you suspect contains
            characters from multiple encodings. Note that this _must_
            be a bytestring. If you've already converted the document
            to Unicode, you're too late.
        :param main_encoding: The primary encoding of `in_bytes`.
        :param embedded_encoding: The encoding that was used to embed characters
            in the main document.
        :return: A bytestring in which `embedded_encoding`
          characters have been converted to their `main_encoding`
          equivalents.
        """
    if embedded_encoding.replace('_', '-').lower() not in ('windows-1252', 'windows_1252'):
        raise NotImplementedError('Windows-1252 and ISO-8859-1 are the only currently supported embedded encodings.')
    if main_encoding.lower() not in ('utf8', 'utf-8'):
        raise NotImplementedError('UTF-8 is the only currently supported main encoding.')
    byte_chunks = []
    chunk_start = 0
    pos = 0
    while pos < len(in_bytes):
        byte = in_bytes[pos]
        if not isinstance(byte, int):
            byte = ord(byte)
        if byte >= cls.FIRST_MULTIBYTE_MARKER and byte <= cls.LAST_MULTIBYTE_MARKER:
            for start, end, size in cls.MULTIBYTE_MARKERS_AND_SIZES:
                if byte >= start and byte <= end:
                    pos += size
                    break
        elif byte >= 128 and byte in cls.WINDOWS_1252_TO_UTF8:
            byte_chunks.append(in_bytes[chunk_start:pos])
            byte_chunks.append(cls.WINDOWS_1252_TO_UTF8[byte])
            pos += 1
            chunk_start = pos
        else:
            pos += 1
    if chunk_start == 0:
        return in_bytes
    else:
        byte_chunks.append(in_bytes[chunk_start:])
    return b''.join(byte_chunks)