from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbx_group(self, objects_dict: PbxDict) -> None:
    groupmap = {}
    target_src_map = {}
    for t in self.build_targets:
        groupmap[t] = self.gen_id()
        target_src_map[t] = self.gen_id()
    for t in self.custom_targets:
        groupmap[t] = self.gen_id()
        target_src_map[t] = self.gen_id()
    projecttree_id = self.gen_id()
    resources_id = self.gen_id()
    products_id = self.gen_id()
    frameworks_id = self.gen_id()
    main_dict = PbxDict()
    objects_dict.add_item(self.maingroup_id, main_dict)
    main_dict.add_item('isa', 'PBXGroup')
    main_children = PbxArray()
    main_dict.add_item('children', main_children)
    main_children.add_item(projecttree_id, 'Project tree')
    main_children.add_item(resources_id, 'Resources')
    main_children.add_item(products_id, 'Products')
    main_children.add_item(frameworks_id, 'Frameworks')
    main_dict.add_item('sourceTree', '"<group>"')
    self.add_projecttree(objects_dict, projecttree_id)
    resource_dict = PbxDict()
    objects_dict.add_item(resources_id, resource_dict, 'Resources')
    resource_dict.add_item('isa', 'PBXGroup')
    resource_children = PbxArray()
    resource_dict.add_item('children', resource_children)
    resource_dict.add_item('name', 'Resources')
    resource_dict.add_item('sourceTree', '"<group>"')
    frameworks_dict = PbxDict()
    objects_dict.add_item(frameworks_id, frameworks_dict, 'Frameworks')
    frameworks_dict.add_item('isa', 'PBXGroup')
    frameworks_children = PbxArray()
    frameworks_dict.add_item('children', frameworks_children)
    for t in self.build_targets.values():
        for dep in t.get_external_deps():
            if dep.name == 'appleframeworks':
                for f in dep.frameworks:
                    frameworks_children.add_item(self.native_frameworks_fileref[f], f)
    frameworks_dict.add_item('name', 'Frameworks')
    frameworks_dict.add_item('sourceTree', '"<group>"')
    for tname, t in self.custom_targets.items():
        target_dict = PbxDict()
        objects_dict.add_item(groupmap[tname], target_dict, tname)
        target_dict.add_item('isa', 'PBXGroup')
        target_children = PbxArray()
        target_dict.add_item('children', target_children)
        target_children.add_item(target_src_map[tname], 'Source files')
        if t.subproject:
            target_dict.add_item('name', f'"{t.subproject} • {t.name}"')
        else:
            target_dict.add_item('name', f'"{t.name}"')
        target_dict.add_item('sourceTree', '"<group>"')
        source_files_dict = PbxDict()
        objects_dict.add_item(target_src_map[tname], source_files_dict, 'Source files')
        source_files_dict.add_item('isa', 'PBXGroup')
        source_file_children = PbxArray()
        source_files_dict.add_item('children', source_file_children)
        for s in t.sources:
            if isinstance(s, mesonlib.File):
                s = os.path.join(s.subdir, s.fname)
            elif isinstance(s, str):
                s = os.path.join(t.subdir, s)
            else:
                continue
            source_file_children.add_item(self.fileref_ids[tname, s], s)
        source_files_dict.add_item('name', '"Source files"')
        source_files_dict.add_item('sourceTree', '"<group>"')
    product_dict = PbxDict()
    objects_dict.add_item(products_id, product_dict, 'Products')
    product_dict.add_item('isa', 'PBXGroup')
    product_children = PbxArray()
    product_dict.add_item('children', product_children)
    for t in self.build_targets:
        product_children.add_item(self.target_filemap[t], t)
    product_dict.add_item('name', 'Products')
    product_dict.add_item('sourceTree', '"<group>"')