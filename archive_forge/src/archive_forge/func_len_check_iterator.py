import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def len_check_iterator(content, stream, content_len=None):
    """Flatten a parser's output into tokens and verify it covers the entire line/text"""
    if content_len is None:
        content_len = len(content)
    covered = 0
    for token_or_element in stream:
        try:
            tokens = cast('Deb822Element', token_or_element).iter_tokens()
        except AttributeError:
            token = cast('Deb822Token', token_or_element)
            covered += len(token.text)
        else:
            for token in tokens:
                covered += len(token.text)
        yield token_or_element
    if covered != content_len:
        if covered < content_len:
            msg = textwrap.dedent('            Value parser did not fully cover the entire line with tokens (\n            missing range {covered}..{content_len}).  Occurred when parsing "{content}"\n            ').format(covered=covered, content_len=content_len, line=content)
            raise ValueError(msg)
        msg = textwrap.dedent('                    Value parser emitted tokens for more text than was present?  Should have\n                     emitted {content_len} characters, got {covered}. Occurred when parsing\n                     "{content}"\n                    ').format(covered=covered, content_len=content_len, content=content)
        raise ValueError(msg)