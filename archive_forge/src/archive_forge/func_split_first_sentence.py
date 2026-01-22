import contextlib
import re
from typing import List, Match, Optional, Union
def split_first_sentence(text):
    """Split text into first sentence and the rest.

    Return a tuple (sentence, rest).
    """
    sentence = ''
    rest = text
    delimiter = ''
    previous_delimiter = ''
    while rest:
        split = re.split('(\\s)', rest, maxsplit=1)
        word = split[0]
        if len(split) == 3:
            delimiter = split[1]
            rest = split[2]
        else:
            assert len(split) == 1
            delimiter = ''
            rest = ''
        sentence += previous_delimiter + word
        if sentence.endswith(('e.g.', 'i.e.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.')):
            pass
        elif sentence.endswith(('.', '?', '!')):
            break
        elif sentence.endswith(':') and delimiter == '\n':
            break
        previous_delimiter = delimiter
        delimiter = ''
    return (sentence, delimiter + rest)