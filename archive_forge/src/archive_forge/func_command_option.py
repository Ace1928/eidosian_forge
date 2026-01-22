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
def command_option(prefix, options):
    arguments = '\n  '.join(options['arguments'])
    return f'{prefix}_options=(\n  {arguments}\n)\n'