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
@panels.setter
def panels(self, new_panels):
    json_path = self._get_path('panels')
    new_specs = [p.spec for p in fix_collisions(new_panels)]
    nested_set(self, json_path, new_specs)