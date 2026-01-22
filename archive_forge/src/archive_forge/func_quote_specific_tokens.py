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
def quote_specific_tokens(tokens: List[str], tokens_to_quote: List[str]) -> None:
    """
    Quote specific tokens in a list

    :param tokens: token list being edited
    :param tokens_to_quote: the tokens, which if present in tokens, to quote
    """
    for i, token in enumerate(tokens):
        if token in tokens_to_quote:
            tokens[i] = quote_string(token)