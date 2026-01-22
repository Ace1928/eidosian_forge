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
def get_option_strings(parser):
    """Flattened list of all `parser`'s option strings."""
    return sum((opt.option_strings for opt in parser._get_optional_actions() if opt.help != SUPPRESS), [])