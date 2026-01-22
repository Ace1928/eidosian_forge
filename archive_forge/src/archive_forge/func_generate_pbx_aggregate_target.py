from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_aggregate_target(self, objects_dict: PbxDict) -> None:
    self.custom_aggregate_targets = {}
    self.build_all_tdep_id = self.gen_id()
    target_dependencies = []
    custom_target_dependencies = []
    for tname, t in self.get_build_by_default_targets().items():
        if isinstance(t, build.CustomTarget):
            custom_target_dependencies.append(self.pbx_custom_dep_map[t.get_id()])
        elif isinstance(t, build.BuildTarget):
            target_dependencies.append(self.pbx_dep_map[t.get_id()])
    aggregated_targets = []
    aggregated_targets.append((self.all_id, 'ALL_BUILD', self.all_buildconf_id, [], [self.regen_dependency_id] + target_dependencies + custom_target_dependencies))
    aggregated_targets.append((self.test_id, 'RUN_TESTS', self.test_buildconf_id, [self.test_command_id], [self.regen_dependency_id, self.build_all_tdep_id]))
    aggregated_targets.append((self.regen_id, 'REGENERATE', self.regen_buildconf_id, [self.regen_command_id], []))
    for tname, t in self.build.get_custom_targets().items():
        ct_id = self.gen_id()
        self.custom_aggregate_targets[tname] = ct_id
        build_phases = []
        dependencies = [self.regen_dependency_id]
        generator_id = 0
        for d in t.dependencies:
            if isinstance(d, build.CustomTarget):
                dependencies.append(self.pbx_custom_dep_map[d.get_id()])
            elif isinstance(d, build.BuildTarget):
                dependencies.append(self.pbx_dep_map[d.get_id()])
        for s in t.sources:
            if isinstance(s, build.GeneratedList):
                build_phases.append(self.shell_targets[tname, generator_id])
                for d in s.depends:
                    dependencies.append(self.pbx_custom_dep_map[d.get_id()])
                generator_id += 1
            elif isinstance(s, build.ExtractedObjects):
                source_target_id = self.pbx_dep_map[s.target.get_id()]
                if source_target_id not in dependencies:
                    dependencies.append(source_target_id)
        build_phases.append(self.shell_targets[tname])
        aggregated_targets.append((ct_id, tname, self.buildconflistmap[tname], build_phases, dependencies))
    sorted_aggregated_targets = sorted(aggregated_targets, key=operator.itemgetter(0))
    for t in sorted_aggregated_targets:
        agt_dict = PbxDict()
        name = t[1]
        buildconf_id = t[2]
        build_phases = t[3]
        dependencies = t[4]
        agt_dict.add_item('isa', 'PBXAggregateTarget')
        agt_dict.add_item('buildConfigurationList', buildconf_id, f'Build configuration list for PBXAggregateTarget "{name}"')
        bp_arr = PbxArray()
        agt_dict.add_item('buildPhases', bp_arr)
        for bp in build_phases:
            bp_arr.add_item(bp, 'ShellScript')
        dep_arr = PbxArray()
        agt_dict.add_item('dependencies', dep_arr)
        for td in dependencies:
            dep_arr.add_item(td, 'PBXTargetDependency')
        agt_dict.add_item('name', f'"{name}"')
        agt_dict.add_item('productName', f'"{name}"')
        objects_dict.add_item(t[0], agt_dict, name)