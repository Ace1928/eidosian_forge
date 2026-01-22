from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_native_target(self, objects_dict: PbxDict) -> None:
    for tname, idval in self.native_targets.items():
        ntarget_dict = PbxDict()
        t = self.build_targets[tname]
        objects_dict.add_item(idval, ntarget_dict, tname)
        ntarget_dict.add_item('isa', 'PBXNativeTarget')
        ntarget_dict.add_item('buildConfigurationList', self.buildconflistmap[tname], f'Build configuration list for PBXNativeTarget "{tname}"')
        buildphases_array = PbxArray()
        ntarget_dict.add_item('buildPhases', buildphases_array)
        generator_id = 0
        for g in t.generated:
            if isinstance(g, build.GeneratedList):
                buildphases_array.add_item(self.shell_targets[tname, generator_id], f'Generator {generator_id}/{tname}')
                generator_id += 1
        for bpname, bpval in t.buildphasemap.items():
            buildphases_array.add_item(bpval, f'{bpname} yyy')
        ntarget_dict.add_item('buildRules', PbxArray())
        dep_array = PbxArray()
        ntarget_dict.add_item('dependencies', dep_array)
        dep_array.add_item(self.regen_dependency_id)
        for lt in self.build_targets[tname].link_targets:
            if isinstance(lt, build.CustomTarget):
                dep_array.add_item(self.pbx_custom_dep_map[lt.get_id()], lt.name)
            elif isinstance(lt, build.CustomTargetIndex):
                dep_array.add_item(self.pbx_custom_dep_map[lt.target.get_id()], lt.target.name)
            else:
                idval = self.pbx_dep_map[lt.get_id()]
                dep_array.add_item(idval, 'PBXTargetDependency')
        for o in t.objects:
            if isinstance(o, build.ExtractedObjects):
                source_target_id = o.target.get_id()
                idval = self.pbx_dep_map[source_target_id]
                dep_array.add_item(idval, 'PBXTargetDependency')
        generator_id = 0
        for o in t.generated:
            if isinstance(o, build.CustomTarget):
                dep_array.add_item(self.pbx_custom_dep_map[o.get_id()], o.name)
            elif isinstance(o, build.CustomTargetIndex):
                dep_array.add_item(self.pbx_custom_dep_map[o.target.get_id()], o.target.name)
            generator_id += 1
        ntarget_dict.add_item('name', f'"{tname}"')
        ntarget_dict.add_item('productName', f'"{tname}"')
        ntarget_dict.add_item('productReference', self.target_filemap[tname], tname)
        if isinstance(t, build.Executable):
            typestr = 'com.apple.product-type.tool'
        elif isinstance(t, build.StaticLibrary):
            typestr = 'com.apple.product-type.library.static'
        elif isinstance(t, build.SharedLibrary):
            typestr = 'com.apple.product-type.library.dynamic'
        else:
            raise MesonException('Unknown target type for %s' % tname)
        ntarget_dict.add_item('productType', f'"{typestr}"')