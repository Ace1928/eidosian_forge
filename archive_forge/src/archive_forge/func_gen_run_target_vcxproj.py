from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def gen_run_target_vcxproj(self, target: build.RunTarget, ofname: str, guid: str) -> None:
    root, type_config = self.create_basic_project(target.name, temp_dir=target.get_id(), guid=guid)
    depend_files = self.get_target_depend_files(target)
    if not target.command:
        assert isinstance(target, build.AliasTarget)
        assert len(depend_files) == 0
    else:
        assert not isinstance(target, build.AliasTarget)
        target_env = self.get_run_target_env(target)
        _, _, cmd_raw = self.eval_custom_target_command(target)
        wrapper_cmd, _ = self.as_meson_exe_cmdline(target.command[0], cmd_raw[1:], force_serialize=True, env=target_env, verbose=True)
        self.add_custom_build(root, 'run_target', ' '.join(self.quote_arguments(wrapper_cmd)), deps=depend_files)
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
    self.add_regen_dependency(root)
    self.add_target_deps(root, target)
    self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)