import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def find_references(module_context, tree_name, only_in_module=False):
    inf = module_context.inference_state
    search_name = tree_name.value
    try:
        inf.flow_analysis_enabled = False
        found_names = _find_defining_names(module_context, tree_name)
    finally:
        inf.flow_analysis_enabled = True
    found_names_dct = _dictionarize(found_names)
    module_contexts = [module_context]
    if not only_in_module:
        for m in set((d.get_root_context() for d in found_names)):
            if m != module_context and m.tree_node is not None and (inf.project.path in m.py__file__().parents):
                module_contexts.append(m)
    if only_in_module or any((n.api_type == 'param' for n in found_names)):
        potential_modules = module_contexts
    else:
        potential_modules = get_module_contexts_containing_name(inf, module_contexts, search_name)
    non_matching_reference_maps = {}
    for module_context in potential_modules:
        for name_leaf in module_context.tree_node.get_used_names().get(search_name, []):
            new = _dictionarize(_find_names(module_context, name_leaf))
            if any((tree_name in found_names_dct for tree_name in new)):
                found_names_dct.update(new)
                for tree_name in new:
                    for dct in non_matching_reference_maps.get(tree_name, []):
                        found_names_dct.update(dct)
                    try:
                        del non_matching_reference_maps[tree_name]
                    except KeyError:
                        pass
            else:
                for name in new:
                    non_matching_reference_maps.setdefault(name, []).append(new)
    result = found_names_dct.values()
    if only_in_module:
        return [n for n in result if n.get_root_context() == module_context]
    return result