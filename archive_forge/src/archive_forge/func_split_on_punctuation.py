import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
def split_on_punctuation(self, tokens: List[str]) -> List[str]:
    """Further splits tokens from a command line using punctuation characters.

        Punctuation characters are treated as word breaks when they are in
        unquoted strings. Each run of punctuation characters is treated as a
        single token.

        :param tokens: the tokens as parsed by shlex
        :return: a new list of tokens, further split using punctuation
        """
    punctuation: List[str] = []
    punctuation.extend(self.terminators)
    punctuation.extend(constants.REDIRECTION_CHARS)
    punctuated_tokens = []
    for cur_initial_token in tokens:
        if len(cur_initial_token) <= 1 or cur_initial_token[0] in constants.QUOTES:
            punctuated_tokens.append(cur_initial_token)
            continue
        cur_index = 0
        cur_char = cur_initial_token[cur_index]
        new_token = ''
        while True:
            if cur_char not in punctuation:
                while cur_char not in punctuation:
                    new_token += cur_char
                    cur_index += 1
                    if cur_index < len(cur_initial_token):
                        cur_char = cur_initial_token[cur_index]
                    else:
                        break
            else:
                cur_punc = cur_char
                while cur_char == cur_punc:
                    new_token += cur_char
                    cur_index += 1
                    if cur_index < len(cur_initial_token):
                        cur_char = cur_initial_token[cur_index]
                    else:
                        break
            punctuated_tokens.append(new_token)
            new_token = ''
            if cur_index >= len(cur_initial_token):
                break
    return punctuated_tokens