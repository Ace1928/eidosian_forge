from __future__ import annotations
import collections
import itertools
import json
import logging
import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING
import requests
from monty.json import MontyDecoder
from pymatgen.core import SETTINGS
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def summary_search(self, **kwargs) -> list[dict]:
    """This function mirrors the mp-api's summary.search functionality.

        Args:
            **kwargs: This function only takes kwargs. All kwargs that do not start with an underscore are treated as
                search criteria and those with underscores are treated as params. Example usage:
                MPRester().summary.search(material_ids="mp-19770,mp-19017", _fields="formula_pretty,energy_above_hull")
        """
    criteria = {k: v for k, v in kwargs.items() if not k.startswith('_')}
    params = [f'{k}={v}' for k, v in kwargs.items() if k.startswith('_') and k != '_fields']
    if '_fields' not in kwargs:
        params.append('_all_fields=True')
    else:
        fields = ','.join(kwargs['_fields']) if isinstance(kwargs['_fields'], list) else kwargs['_fields']
        params.extend((f'_fields={fields}', '_all_fields=False'))
    get = '&'.join(params)
    logger.info(f'query={get}')
    return self.request(f'materials/summary/?{get}', payload=criteria)