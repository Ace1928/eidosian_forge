import contextlib
import re
from typing import List, Match, Optional, Union
def split_summary_and_description(contents):
    """Split docstring into summary and description.

    Return tuple (summary, description).
    """
    split_lines = contents.rstrip().splitlines()
    for index in range(1, len(split_lines)):
        if not split_lines[index].strip() or (index + 1 < len(split_lines) and is_probably_beginning_of_sentence(split_lines[index + 1])):
            return ('\n'.join(split_lines[:index]).strip(), '\n'.join(split_lines[index:]).rstrip())
    split = split_first_sentence(contents)
    if split[0].strip() and split[1].strip():
        return (split[0].strip(), find_shortest_indentation(split[1].splitlines()[1:]) + split[1].strip())
    return (contents, '')