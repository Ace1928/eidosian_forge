import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def complete_subcommand_help(self, text: str, line: str, begidx: int, endidx: int, tokens: List[str]) -> List[str]:
    """
        Supports cmd2's help command in the completion of subcommand names
        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param tokens: arguments passed to command/subcommand
        :return: List of subcommand completions
        """
    if self._subcommand_action is not None:
        for token_index, token in enumerate(tokens):
            if token in self._subcommand_action.choices:
                parser: argparse.ArgumentParser = self._subcommand_action.choices[token]
                completer_type = self._cmd2_app._determine_ap_completer_type(parser)
                completer = completer_type(parser, self._cmd2_app)
                return completer.complete_subcommand_help(text, line, begidx, endidx, tokens[token_index + 1:])
            elif token_index == len(tokens) - 1:
                return self._cmd2_app.basic_complete(text, line, begidx, endidx, self._subcommand_action.choices)
            else:
                break
    return []