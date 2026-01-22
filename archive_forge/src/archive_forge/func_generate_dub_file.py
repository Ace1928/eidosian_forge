from __future__ import annotations
import json
import os
from . import ExtensionModule, ModuleInfo
from .. import mlog
from ..dependencies import Dependency
from ..dependencies.dub import DubDependency
from ..interpreterbase import typed_pos_args
from ..mesonlib import Popen_safe, MesonException, listify
@typed_pos_args('dlang.generate_dub_file', str, str)
def generate_dub_file(self, state, args, kwargs):
    if not DlangModule.init_dub:
        self._init_dub(state)
    config = {'name': args[0]}
    config_path = os.path.join(args[1], 'dub.json')
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as ofile:
            try:
                config = json.load(ofile)
            except ValueError:
                mlog.warning('Failed to load the data in dub.json')
    warn_publishing = ['description', 'license']
    for arg in warn_publishing:
        if arg not in kwargs and arg not in config:
            mlog.warning('Without', mlog.bold(arg), "the DUB package can't be published")
    for key, value in kwargs.items():
        if key == 'dependencies':
            values = listify(value, flatten=False)
            config[key] = {}
            for dep in values:
                if isinstance(dep, Dependency):
                    name = dep.get_name()
                    ret, res = self._call_dubbin(['describe', name])
                    if ret == 0:
                        version = dep.get_version()
                        if version is None:
                            config[key][name] = ''
                        else:
                            config[key][name] = version
        else:
            config[key] = value
    with open(config_path, 'w', encoding='utf-8') as ofile:
        ofile.write(json.dumps(config, indent=4, ensure_ascii=False))