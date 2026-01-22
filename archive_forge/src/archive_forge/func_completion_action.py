import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
def completion_action(parent: Opt[ArgumentParser]=None, preamble: Union[str, Dict[str, str]]=''):

    class PrintCompletionAction(_ShtabPrintCompletionAction):

        def __call__(self, parser, namespace, values, option_string=None):
            print(complete(parent or parser, values, preamble=preamble))
            parser.exit(0)
    return PrintCompletionAction