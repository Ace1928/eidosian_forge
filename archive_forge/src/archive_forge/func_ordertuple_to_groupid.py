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
def ordertuple_to_groupid(ordertuple):
    rs_name, rest = (ordertuple[0], ordertuple[1:])
    rs = self._get_rs_by_name(rs_name)
    id = rs.spec['id']
    keys = [rs.pm_query_generator.pc_front_to_back(k) for k in rs.groupby]
    kvs = [f'{k}:{v}' for k, v in zip(keys, rest)]
    linked = '-'.join(kvs)
    return f'{id}-{linked}'