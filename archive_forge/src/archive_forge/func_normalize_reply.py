import os
import sys as _sys
def normalize_reply(text: str, version=1) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.

    Version 1: Fix sentence start casing, and punctuation.

    Version 2: Add trailing period, if missing.
    """
    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]
    new_text = text.lower()
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '
    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')
    if version > 1 and new_text and (new_text[-1] not in '!.?)"\''):
        new_text += '.'
    return new_text