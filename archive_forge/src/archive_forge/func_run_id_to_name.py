import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
def run_id_to_name(id):
    for rs in self.runsets:
        try:
            run = PublicApi().run(f'{rs.entity}/{rs.project}/{id}')
        except Exception:
            pass
        else:
            return run.name
    raise ValueError('Unable to find this run!')