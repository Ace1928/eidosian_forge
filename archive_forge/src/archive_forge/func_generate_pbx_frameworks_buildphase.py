from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_frameworks_buildphase(self, objects_dict: PbxDict) -> None:
    for t in self.build_targets.values():
        bt_dict = PbxDict()
        objects_dict.add_item(t.buildphasemap['Frameworks'], bt_dict, 'Frameworks')
        bt_dict.add_item('isa', 'PBXFrameworksBuildPhase')
        bt_dict.add_item('buildActionMask', 2147483647)
        file_list = PbxArray()
        bt_dict.add_item('files', file_list)
        for dep in t.get_external_deps():
            if dep.name == 'appleframeworks':
                for f in dep.frameworks:
                    file_list.add_item(self.native_frameworks[f], f'{f}.framework in Frameworks')
        bt_dict.add_item('runOnlyForDeploymentPostprocessing', 0)