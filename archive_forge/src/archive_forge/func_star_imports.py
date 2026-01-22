import os
from pathlib import Path
from typing import Optional
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import AbstractNameDefinition, ModuleName
from jedi.inference.filters import GlobalNameFilter, ParserTreeFilter, DictFilter, MergedFilter
from jedi.inference import compiled
from jedi.inference.base_value import TreeValue
from jedi.inference.names import SubModuleName
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.compiled import create_simple_object
from jedi.inference.base_value import ValueSet
from jedi.inference.context import ModuleContext
@inference_state_method_cache([])
def star_imports(self):
    from jedi.inference.imports import Importer
    modules = []
    module_context = self.as_context()
    for i in self.tree_node.iter_imports():
        if i.is_star_import():
            new = Importer(self.inference_state, import_path=i.get_paths()[-1], module_context=module_context, level=i.level).follow()
            for module in new:
                if isinstance(module, ModuleValue):
                    modules += module.star_imports()
            modules += new
    return modules