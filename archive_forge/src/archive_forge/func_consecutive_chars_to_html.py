import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@staticmethod
def consecutive_chars_to_html(consecutive_chars_list: List[CharState], text: str, encoding: Encoding):
    """
        Converts a list of "consecutive chars" into a single HTML element.
        Chars are consecutive if they fall under the same word, token and annotation.
        The CharState class is a named tuple with a "partition_key" method that makes it easy to
        compare if two chars are consecutive.

        Args:
            consecutive_chars_list (:obj:`List[CharState]`):
                A list of CharStates that have been grouped together

            text (:obj:`str`):
                The original text being processed

            encoding (:class:`~tokenizers.Encoding`):
                The encoding returned from the tokenizer

        Returns:
            :obj:`str`: The HTML span for a set of consecutive chars
        """
    first = consecutive_chars_list[0]
    if first.char_ix is None:
        stoken = encoding.tokens[first.token_ix]
        return f'<span class="special-token" data-stoken={stoken}></span>'
    last = consecutive_chars_list[-1]
    start = first.char_ix
    end = last.char_ix + 1
    span_text = text[start:end]
    css_classes = []
    data_items = {}
    if first.token_ix is not None:
        css_classes.append('token')
        if first.is_multitoken:
            css_classes.append('multi-token')
        if first.token_ix % 2:
            css_classes.append('odd-token')
        else:
            css_classes.append('even-token')
        if EncodingVisualizer.unk_token_regex.search(encoding.tokens[first.token_ix]) is not None:
            css_classes.append('special-token')
            data_items['stok'] = encoding.tokens[first.token_ix]
    else:
        css_classes.append('non-token')
    css = f'class="{' '.join(css_classes)}"'
    data = ''
    for key, val in data_items.items():
        data += f' data-{key}="{val}"'
    return f'<span {css} {data} >{span_text}</span>'