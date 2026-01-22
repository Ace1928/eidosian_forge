import itertools
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import srsly
from wasabi import Printer
from ..tokens import Doc, DocBin
from ..training import docs_to_json
from ..training.converters import (
from ._util import Arg, Opt, app, walk_directory
def autodetect_ner_format(input_data: str) -> Optional[str]:
    lines = input_data.split('\n')[:20]
    format_guesses = {'ner': 0, 'iob': 0}
    iob_re = re.compile('\\S+\\|(O|[IB]-\\S+)')
    ner_re = re.compile('\\S+\\s+(O|[IB]-\\S+)$')
    for line in lines:
        line = line.strip()
        if iob_re.search(line):
            format_guesses['iob'] += 1
        if ner_re.search(line):
            format_guesses['ner'] += 1
    if format_guesses['iob'] == 0 and format_guesses['ner'] > 0:
        return 'ner'
    if format_guesses['ner'] == 0 and format_guesses['iob'] > 0:
        return 'iob'
    return None