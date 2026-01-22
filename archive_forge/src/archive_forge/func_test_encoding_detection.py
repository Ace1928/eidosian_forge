from __future__ import with_statement
import textwrap
from difflib import ndiff
from io import open
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
import pytest
import chardet
@pytest.mark.parametrize('file_name, encoding', gen_test_params())
def test_encoding_detection(file_name, encoding):
    with open(file_name, 'rb') as f:
        input_bytes = f.read()
        result = chardet.detect(input_bytes)
        try:
            expected_unicode = input_bytes.decode(encoding)
        except LookupError:
            expected_unicode = ''
        try:
            detected_unicode = input_bytes.decode(result['encoding'])
        except (LookupError, UnicodeDecodeError, TypeError):
            detected_unicode = ''
    if result:
        encoding_match = (result['encoding'] or '').lower() == encoding
    else:
        encoding_match = False
    if not encoding_match and expected_unicode != detected_unicode:
        wrapped_expected = '\n'.join(textwrap.wrap(expected_unicode, 100)) + '\n'
        wrapped_detected = '\n'.join(textwrap.wrap(detected_unicode, 100)) + '\n'
        diff = ''.join(ndiff(wrapped_expected.splitlines(True), wrapped_detected.splitlines(True)))
    else:
        diff = ''
        encoding_match = True
    assert encoding_match, 'Expected %s, but got %s for %s.  Character differences: \n%s' % (encoding, result, file_name, diff)