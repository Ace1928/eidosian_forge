from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
def find_package_target(pkg: T.Dict[str, str]) -> bool:
    nonlocal show_buildtype_warning
    pack_id = f'{pkg['name']}@{pkg['version']}'
    tgt_file, compatibilities = self._find_compatible_package_target(description, pkg, dub_comp_id)
    if tgt_file is None:
        if not compatibilities:
            mlog.error(mlog.bold(pack_id), 'not found')
        elif 'compiler' not in compatibilities:
            mlog.error(mlog.bold(pack_id), 'found but not compiled with ', mlog.bold(dub_comp_id))
        elif dub_comp_id != 'gdc' and 'compiler_version' not in compatibilities:
            mlog.error(mlog.bold(pack_id), 'found but not compiled with', mlog.bold(f'{dub_comp_id}-{self.compiler.version}'))
        elif 'arch' not in compatibilities:
            mlog.error(mlog.bold(pack_id), 'found but not compiled for', mlog.bold(dub_arch))
        elif 'platform' not in compatibilities:
            mlog.error(mlog.bold(pack_id), 'found but not compiled for', mlog.bold(description['platform'].join('.')))
        elif 'configuration' not in compatibilities:
            mlog.error(mlog.bold(pack_id), 'found but not compiled for the', mlog.bold(pkg['configuration']), 'configuration')
        else:
            mlog.error(mlog.bold(pack_id), 'not found')
        mlog.log('You may try the following command to install the necessary DUB libraries:')
        mlog.log(mlog.bold(dub_build_deep_command()))
        return False
    if 'build_type' not in compatibilities:
        mlog.warning(mlog.bold(pack_id), 'found but not compiled as', mlog.bold(dub_buildtype))
        show_buildtype_warning = True
    self.link_args.append(tgt_file)
    return True