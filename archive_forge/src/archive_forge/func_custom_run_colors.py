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
@custom_run_colors.setter
def custom_run_colors(self, new_custom_run_colors):
    json_path = self._get_path('custom_run_colors')
    color_settings = {}

    def ordertuple_to_groupid(ordertuple):
        rs_name, rest = (ordertuple[0], ordertuple[1:])
        rs = self._get_rs_by_name(rs_name)
        id = rs.spec['id']
        keys = [rs.pm_query_generator.pc_front_to_back(k) for k in rs.groupby]
        kvs = [f'{k}:{v}' for k, v in zip(keys, rest)]
        linked = '-'.join(kvs)
        return f'{id}-{linked}'

    def run_name_to_id(name):
        for rs in self.runsets:
            runs = PublicApi().runs(path=f'{rs.entity}/{rs.project}', filters={'display_name': name})
            if len(runs) > 1:
                termwarn('Multiple runs with the same name found! Using the first one.')
            for run in runs:
                if run.name == name:
                    return run.id
        raise ValueError('Unable to find this run!')
    for name, c in new_custom_run_colors.items():
        if isinstance(name, tuple):
            key = ordertuple_to_groupid(name)
        else:
            key = run_name_to_id(name)
        color_settings[key] = c
    nested_set(self, json_path, color_settings)