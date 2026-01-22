from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def generate_depmf_install(self, d: InstallData) -> None:
    depmf_path = self.build.dep_manifest_name
    if depmf_path is None:
        option_dir = self.environment.coredata.get_option(OptionKey('licensedir'))
        assert isinstance(option_dir, str), 'for mypy'
        if option_dir:
            depmf_path = os.path.join(option_dir, 'depmf.json')
        else:
            return
    ifilename = os.path.join(self.environment.get_build_dir(), 'depmf.json')
    ofilename = os.path.join(self.environment.get_prefix(), depmf_path)
    odirname = os.path.join(self.environment.get_prefix(), os.path.dirname(depmf_path))
    out_name = os.path.join('{prefix}', depmf_path)
    out_dir = os.path.join('{prefix}', os.path.dirname(depmf_path))
    mfobj = {'type': 'dependency manifest', 'version': '1.0', 'projects': {k: v.to_json() for k, v in self.build.dep_manifest.items()}}
    with open(ifilename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(mfobj))
    d.data.append(InstallDataBase(ifilename, ofilename, out_name, None, '', tag='devel', data_type='depmf'))
    for m in self.build.dep_manifest.values():
        for ifilename, name in m.license_files:
            ofilename = os.path.join(odirname, name.relative_name())
            out_name = os.path.join(out_dir, name.relative_name())
            d.data.append(InstallDataBase(ifilename, ofilename, out_name, None, m.subproject, tag='devel', data_type='depmf'))