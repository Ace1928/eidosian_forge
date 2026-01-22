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
def parse_and_process_known_args(self, args=None):
    """
        Parse provided arguments and return parlai opts and unknown arg list.

        Runs the same arg->opt parsing that parse_args does, but doesn't throw an error
        if the args being parsed include additional command line arguments that parlai
        doesn't know what to do with.
        """
    self.args, unknowns = super().parse_known_args(args=args)
    self._process_args_to_opts(args)
    return (self.opt, unknowns)