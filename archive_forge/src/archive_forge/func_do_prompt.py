import argparse
import locale
import os
import sys
import time
from collections import OrderedDict
from os import path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from docutils.utils import column_width
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.locale import __
from sphinx.util.console import bold, color_terminal, colorize, nocolor, red  # type: ignore
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer
def do_prompt(text: str, default: Optional[str]=None, validator: Callable[[str], Any]=nonempty) -> Union[str, bool]:
    while True:
        if default is not None:
            prompt = PROMPT_PREFIX + '%s [%s]: ' % (text, default)
        else:
            prompt = PROMPT_PREFIX + text + ': '
        if USE_LIBEDIT:
            pass
        elif READLINE_AVAILABLE:
            prompt = colorize(COLOR_QUESTION, prompt, input_mode=True)
        else:
            prompt = colorize(COLOR_QUESTION, prompt, input_mode=False)
        x = term_input(prompt).strip()
        if default and (not x):
            x = default
        try:
            x = validator(x)
        except ValidationError as err:
            print(red('* ' + str(err)))
            continue
        break
    return x