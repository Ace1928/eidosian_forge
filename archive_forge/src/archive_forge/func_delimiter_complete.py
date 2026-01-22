import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def delimiter_complete(self, text: str, line: str, begidx: int, endidx: int, match_against: Iterable[str], delimiter: str) -> List[str]:
    """
        Performs tab completion against a list but each match is split on a delimiter and only
        the portion of the match being tab completed is shown as the completion suggestions.
        This is useful if you match against strings that are hierarchical in nature and have a
        common delimiter.

        An easy way to illustrate this concept is path completion since paths are just directories/files
        delimited by a slash. If you are tab completing items in /home/user you don't get the following
        as suggestions:

        /home/user/file.txt     /home/user/program.c
        /home/user/maps/        /home/user/cmd2.py

        Instead you are shown:

        file.txt                program.c
        maps/                   cmd2.py

        For a large set of data, this can be visually more pleasing and easier to search.

        Another example would be strings formatted with the following syntax: company::department::name
        In this case the delimiter would be :: and the user could easily narrow down what they are looking
        for if they were only shown suggestions in the category they are at in the string.

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param match_against: the list being matched against
        :param delimiter: what delimits each portion of the matches (ex: paths are delimited by a slash)
        :return: a list of possible tab completions
        """
    matches = self.basic_complete(text, line, begidx, endidx, match_against)
    if matches:
        self.matches_delimited = True
        common_prefix = os.path.commonprefix(matches)
        prefix_tokens = common_prefix.split(delimiter)
        display_token_index = 0
        if prefix_tokens:
            display_token_index = len(prefix_tokens) - 1
        for cur_match in matches:
            match_tokens = cur_match.split(delimiter)
            display_token = match_tokens[display_token_index]
            if not display_token:
                display_token = delimiter
            self.display_matches.append(display_token)
    return matches