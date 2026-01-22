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
def get_specials(arg, arg_type, arg_sel):
    if arg.choices:
        choice_strs = ' '.join(map(str, arg.choices))
        yield f"'{arg_type}/{arg_sel}/({choice_strs})/'"
    elif hasattr(arg, 'complete'):
        complete_fn = complete2pattern(arg.complete, 'tcsh', choice_type2fn)
        if complete_fn:
            yield f"'{arg_type}/{arg_sel}/{complete_fn}/'"