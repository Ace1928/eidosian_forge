from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
def make_legal_content(content):
    """
    Remove illegal content from a content block. Illegal content includes:

    * Blank lines
    * Starting or ending with a blank line

    .. doctest::

        >>> make_legal_content('\\nfoo\\n\\nbar\\n')
        'foo\\nbar'

    :param str content: The content to make legal
    :returns: The legalised content
    :rtype: srt
    """
    if content and content[0] != '\n' and ('\n\n' not in content):
        return content
    legal_content = MULTI_WS_REGEX.sub('\n', content.strip('\n'))
    LOG.info('Legalised content %r to %r', content, legal_content)
    return legal_content