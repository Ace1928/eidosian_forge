import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def class2str(value):
    """
    Inverse of params.str2class().
    """
    s = str(value)
    s = s[s.find("'") + 1:s.rfind("'")]
    s = ':'.join(s.rsplit('.', 1))
    return s