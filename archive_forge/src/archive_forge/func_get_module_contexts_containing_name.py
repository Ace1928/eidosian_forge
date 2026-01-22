import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def get_module_contexts_containing_name(inference_state, module_contexts, name, limit_reduction=1):
    """
    Search a name in the directories of modules.

    :param limit_reduction: Divides the limits on opening/parsing files by this
        factor.
    """
    for module_context in module_contexts:
        if module_context.is_compiled():
            continue
        yield module_context
    if len(name) <= 2:
        return
    file_io_iterator = _find_project_modules(inference_state, module_contexts)
    yield from search_in_file_ios(inference_state, file_io_iterator, name, limit_reduction=limit_reduction)