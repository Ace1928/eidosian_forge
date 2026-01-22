import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def unquote_specific_tokens(tokens: List[str], tokens_to_unquote: List[str]) -> None:
    """
    Unquote specific tokens in a list

    :param tokens: token list being edited
    :param tokens_to_unquote: the tokens, which if present in tokens, to unquote
    """
    for i, token in enumerate(tokens):
        unquoted_token = strip_quotes(token)
        if unquoted_token in tokens_to_unquote:
            tokens[i] = unquoted_token