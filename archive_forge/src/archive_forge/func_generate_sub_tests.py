from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
def generate_sub_tests(cls, module, markers):
    if 'backend' in markers or 'sparse_backend' in markers:
        sparse = 'sparse_backend' in markers
        for cfg in _possible_configs_for_cls(cls, sparse=sparse):
            orig_name = cls.__name__
            alpha_name = re.sub('[_\\[\\]\\.]+', '_', cfg.name)
            alpha_name = re.sub('_+$', '', alpha_name)
            name = '%s_%s' % (cls.__name__, alpha_name)
            subcls = type(name, (cls,), {'_sa_orig_cls_name': orig_name, '__only_on_config__': cfg})
            setattr(module, name, subcls)
            yield subcls
    else:
        yield cls