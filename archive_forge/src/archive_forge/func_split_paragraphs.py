import numbers
import random
import re
import string
import textwrap
def split_paragraphs(text):
    """
    Split a string into paragraphs (one or more lines delimited by an empty line).

    :param text: The text to split into paragraphs (a string).
    :returns: A list of strings.
    """
    paragraphs = []
    for chunk in text.split('\n\n'):
        chunk = trim_empty_lines(chunk)
        if chunk and (not chunk.isspace()):
            paragraphs.append(chunk)
    return paragraphs