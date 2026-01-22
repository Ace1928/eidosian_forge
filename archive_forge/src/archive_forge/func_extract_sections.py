from copy import deepcopy
from abc import ABC, abstractmethod
from types import ModuleType
from typing import (
import sys
import token, tokenize
import os
from os import path
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
import lark
from lark.tools import lalr_argparser, build_lalr, make_warnings_comments
from lark.grammar import Rule
from lark.lexer import TerminalDef
def extract_sections(lines):
    section = None
    text = []
    sections = defaultdict(list)
    for line in lines:
        if line.startswith('###'):
            if line[3] == '{':
                section = line[4:].strip()
            elif line[3] == '}':
                sections[section] += text
                section = None
                text = []
            else:
                raise ValueError(line)
        elif section:
            text.append(line)
    return {name: ''.join(text) for name, text in sections.items()}