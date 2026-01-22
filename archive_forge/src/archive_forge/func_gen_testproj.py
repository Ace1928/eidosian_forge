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
def gen_testproj(self):
    project_name = 'RUN_TESTS'
    ofname = os.path.join(self.environment.get_build_dir(), f'{project_name}.vcxproj')
    guid = self.environment.coredata.test_guid
    if self.gen_lite:
        root, type_config = self.create_basic_project(project_name, temp_dir='install-temp', guid=guid, conftype='Makefile')
        nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
        multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
        _, build_dir_tail = os.path.split(self.src_to_build)
        proj_to_multiconfigured_builds_parent_dir = '..'
        for buildtype in multi_config_buildtype_list:
            meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
            proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
            test_cmd = f'{nmake_base_meson_command} test -C "{proj_to_build_dir_for_buildtype}" --no-rebuild'
            if not self.environment.coredata.get_option(OptionKey('stdsplit')):
                test_cmd += ' --no-stdsplit'
            if self.environment.coredata.get_option(OptionKey('errorlogs')):
                test_cmd += ' --print-errorlogs'
            condition = f"'$(Configuration)|$(Platform)'=='{buildtype}|{self.platform}'"
            prop_group = ET.SubElement(root, 'PropertyGroup', Condition=condition)
            ET.SubElement(prop_group, 'NMakeBuildCommandLine').text = test_cmd
            ET.SubElement(prop_group, 'ExecutablePath').text = exe_search_paths
    else:
        root, type_config = self.create_basic_project(project_name, temp_dir='test-temp', guid=guid)
        action = ET.SubElement(root, 'ItemDefinitionGroup')
        midl = ET.SubElement(action, 'Midl')
        ET.SubElement(midl, 'AdditionalIncludeDirectories').text = '%(AdditionalIncludeDirectories)'
        ET.SubElement(midl, 'OutputDirectory').text = '$(IntDir)'
        ET.SubElement(midl, 'HeaderFileName').text = '%(Filename).h'
        ET.SubElement(midl, 'TypeLibraryName').text = '%(Filename).tlb'
        ET.SubElement(midl, 'InterfaceIdentifierFilename').text = '%(Filename)_i.c'
        ET.SubElement(midl, 'ProxyFileName').text = '%(Filename)_p.c'
        test_command = self.environment.get_build_command() + ['test', '--no-rebuild']
        if not self.environment.coredata.get_option(OptionKey('stdsplit')):
            test_command += ['--no-stdsplit']
        if self.environment.coredata.get_option(OptionKey('errorlogs')):
            test_command += ['--print-errorlogs']
        self.serialize_tests()
        self.add_custom_build(root, 'run_tests', '"%s"' % '" "'.join(test_command))
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
    self.add_regen_dependency(root)
    self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)