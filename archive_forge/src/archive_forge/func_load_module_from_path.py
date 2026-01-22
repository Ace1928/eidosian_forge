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
def load_module_from_path(inference_state, file_io, import_names=None, is_package=None):
    """
    This should pretty much only be used for get_modules_containing_name. It's
    here to ensure that a random path is still properly loaded into the Jedi
    module structure.
    """
    path = Path(file_io.path)
    if import_names is None:
        e_sys_path = inference_state.get_sys_path()
        import_names, is_package = sys_path.transform_path_to_dotted(e_sys_path, path)
    else:
        assert isinstance(is_package, bool)
    is_stub = path.suffix == '.pyi'
    if is_stub:
        folder_io = file_io.get_parent_folder()
        if folder_io.path.endswith('-stubs'):
            folder_io = FolderIO(folder_io.path[:-6])
        if path.name == '__init__.pyi':
            python_file_io = folder_io.get_file_io('__init__.py')
        else:
            python_file_io = folder_io.get_file_io(import_names[-1] + '.py')
        try:
            v = load_module_from_path(inference_state, python_file_io, import_names, is_package=is_package)
            values = ValueSet([v])
        except FileNotFoundError:
            values = NO_VALUES
        return create_stub_module(inference_state, inference_state.latest_grammar, values, parse_stub_module(inference_state, file_io), file_io, import_names)
    else:
        module = _load_python_module(inference_state, file_io, import_names=import_names, is_package=is_package)
        inference_state.module_cache.add(import_names, ValueSet([module]))
        return module