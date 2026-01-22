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
def generate_solution(self, sln_filename: str, projlist: T.List[Project]) -> None:
    default_projlist = self.get_build_by_default_targets()
    default_projlist.update(self.get_testlike_targets())
    sln_filename_tmp = sln_filename + '~'
    with open(sln_filename_tmp, 'w', encoding='utf-8-sig') as ofile:
        ofile.write('\nMicrosoft Visual Studio Solution File, Format Version %s\n' % self.sln_file_version)
        ofile.write('# Visual Studio %s\n' % self.sln_version_comment)
        prj_templ = 'Project("{%s}") = "%s", "%s", "{%s}"\n'
        for prj in projlist:
            if self.environment.coredata.get_option(OptionKey('layout')) == 'mirror':
                self.generate_solution_dirs(ofile, prj[1].parents)
            target = self.build.targets[prj[0]]
            lang = 'default'
            if hasattr(target, 'compilers') and target.compilers:
                for lang_out in target.compilers.keys():
                    lang = lang_out
                    break
            prj_line = prj_templ % (self.environment.coredata.lang_guids[lang], prj[0], prj[1], prj[2])
            ofile.write(prj_line)
            target_dict = {target.get_id(): target}
            recursive_deps = self.get_target_deps(target_dict, recursive=True)
            ofile.write('EndProject\n')
            for dep, target in recursive_deps.items():
                if prj[0] in default_projlist:
                    default_projlist[dep] = target
        test_line = prj_templ % (self.environment.coredata.lang_guids['default'], 'RUN_TESTS', 'RUN_TESTS.vcxproj', self.environment.coredata.test_guid)
        ofile.write(test_line)
        ofile.write('EndProject\n')
        if self.gen_lite:
            regen_proj_name = 'RECONFIGURE'
            regen_proj_fname = 'RECONFIGURE.vcxproj'
        else:
            regen_proj_name = 'REGEN'
            regen_proj_fname = 'REGEN.vcxproj'
        regen_line = prj_templ % (self.environment.coredata.lang_guids['default'], regen_proj_name, regen_proj_fname, self.environment.coredata.regen_guid)
        ofile.write(regen_line)
        ofile.write('EndProject\n')
        install_line = prj_templ % (self.environment.coredata.lang_guids['default'], 'RUN_INSTALL', 'RUN_INSTALL.vcxproj', self.environment.coredata.install_guid)
        ofile.write(install_line)
        ofile.write('EndProject\n')
        ofile.write('Global\n')
        ofile.write('\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\n')
        multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list() if self.gen_lite else [self.buildtype]
        for buildtype in multi_config_buildtype_list:
            ofile.write('\t\t%s|%s = %s|%s\n' % (buildtype, self.platform, buildtype, self.platform))
        ofile.write('\tEndGlobalSection\n')
        ofile.write('\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n')
        for buildtype in multi_config_buildtype_list:
            ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.regen_guid, buildtype, self.platform, buildtype, self.platform))
            if not self.gen_lite:
                ofile.write('\t\t{%s}.%s|%s.Build.0 = %s|%s\n' % (self.environment.coredata.regen_guid, buildtype, self.platform, buildtype, self.platform))
        for project_index, p in enumerate(projlist):
            if p[3] is MachineChoice.BUILD:
                config_platform = self.build_platform
            else:
                config_platform = self.platform
            for buildtype in multi_config_buildtype_list:
                ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (p[2], buildtype, self.platform, buildtype, config_platform))
                if (not self.gen_lite or project_index == 0) and p[0] in default_projlist and (not isinstance(self.build.targets[p[0]], build.RunTarget)):
                    ofile.write('\t\t{%s}.%s|%s.Build.0 = %s|%s\n' % (p[2], buildtype, self.platform, buildtype, config_platform))
        for buildtype in multi_config_buildtype_list:
            ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.test_guid, buildtype, self.platform, buildtype, self.platform))
            ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.install_guid, buildtype, self.platform, buildtype, self.platform))
        ofile.write('\tEndGlobalSection\n')
        ofile.write('\tGlobalSection(SolutionProperties) = preSolution\n')
        ofile.write('\t\tHideSolutionNode = FALSE\n')
        ofile.write('\tEndGlobalSection\n')
        if self.subdirs:
            ofile.write('\tGlobalSection(NestedProjects) = preSolution\n')
            for p in projlist:
                if p[1].parent != PurePath('.'):
                    ofile.write('\t\t{{{}}} = {{{}}}\n'.format(p[2], self.subdirs[p[1].parent][0]))
            for subdir in self.subdirs.values():
                if subdir[1]:
                    ofile.write('\t\t{{{}}} = {{{}}}\n'.format(subdir[0], subdir[1]))
            ofile.write('\tEndGlobalSection\n')
        ofile.write('EndGlobal\n')
    replace_if_different(sln_filename, sln_filename_tmp)