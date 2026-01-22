import os
from pathlib import Path
from parso.python import tree
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.file_io import FolderIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference import sys_path
from jedi.inference import helpers
from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.utils import unite
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.typeshed import import_module_decorator, \
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.plugins import plugin_manager
def completion_names(self, inference_state, only_modules=False):
    """
        :param only_modules: Indicates wheter it's possible to import a
            definition that is not defined in a module.
        """
    if not self._infer_possible:
        return []
    names = []
    if self.import_path:
        if self._str_import_path == ('flask', 'ext'):
            for mod in self._get_module_names():
                modname = mod.string_name
                if modname.startswith('flask_'):
                    extname = modname[len('flask_'):]
                    names.append(ImportName(self._module_context, extname))
            for dir in self._sys_path_with_modifications(is_completion=True):
                flaskext = os.path.join(dir, 'flaskext')
                if os.path.isdir(flaskext):
                    names += self._get_module_names([flaskext])
        values = self.follow()
        for value in values:
            if value.api_type not in ('module', 'namespace'):
                continue
            if not value.is_compiled():
                names += value.sub_modules_dict().values()
        if not only_modules:
            from jedi.inference.gradual.conversion import convert_values
            both_values = values | convert_values(values)
            for c in both_values:
                for filter in c.get_filters():
                    names += filter.values()
    elif self.level:
        names += self._get_module_names(self._fixed_sys_path)
    else:
        names += self._get_module_names()
    return names