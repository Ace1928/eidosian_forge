import collections
from typing import Optional
import regex
import tiktoken
def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f'\x1b[48;5;{i}m' for i in [167, 179, 185, 77, 80, 68, 134]]
    unicode_token_values = [x.decode('utf-8', errors='replace') for x in token_values]
    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end='')
    print('\x1b[0m')