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
def load_namespace_from_path(inference_state, folder_io):
    import_names, is_package = sys_path.transform_path_to_dotted(inference_state.get_sys_path(), Path(folder_io.path))
    from jedi.inference.value.namespace import ImplicitNamespaceValue
    return ImplicitNamespaceValue(inference_state, import_names, [folder_io.path])