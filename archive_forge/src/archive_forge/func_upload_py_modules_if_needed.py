import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.working_dir import set_pythonpath_in_context
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
def upload_py_modules_if_needed(runtime_env: Dict[str, Any], scratch_dir: Optional[str]=os.getcwd(), logger: Optional[logging.Logger]=default_logger, upload_fn=None) -> Dict[str, Any]:
    """Uploads the entries in py_modules and replaces them with a list of URIs.

    For each entry that is already a URI, this is a no-op.
    """
    py_modules = runtime_env.get('py_modules')
    if py_modules is None:
        return runtime_env
    if not isinstance(py_modules, list):
        raise TypeError(f'py_modules must be a List of local paths, imported modules, or URIs, got {type(py_modules)}.')
    py_modules_uris = []
    for module in py_modules:
        if isinstance(module, str):
            module_path = module
        elif isinstance(module, Path):
            module_path = str(module)
        elif isinstance(module, ModuleType):
            if len(module.__path__) > 1:
                raise ValueError('py_modules only supports modules whose __path__ has length 1.')
            [module_path] = module.__path__
        else:
            raise TypeError(f'py_modules must be a list of file paths, URIs, or imported modules, got {type(module)}.')
        if _check_is_uri(module_path):
            module_uri = module_path
        elif Path(module_path).is_dir():
            excludes = runtime_env.get('excludes', None)
            module_uri = get_uri_for_directory(module_path, excludes=excludes)
            if upload_fn is None:
                try:
                    upload_package_if_needed(module_uri, scratch_dir, module_path, excludes=excludes, include_parent_dir=True, logger=logger)
                except Exception as e:
                    raise RuntimeEnvSetupError(f'Failed to upload module {module_path} to the Ray cluster: {e}') from e
            else:
                upload_fn(module_path, excludes=excludes)
        elif Path(module_path).suffix == '.whl':
            module_uri = get_uri_for_package(Path(module_path))
            if upload_fn is None:
                if not package_exists(module_uri):
                    try:
                        upload_package_to_gcs(module_uri, Path(module_path).read_bytes())
                    except Exception as e:
                        raise RuntimeEnvSetupError(f'Failed to upload {module_path} to the Ray cluster: {e}') from e
            else:
                upload_fn(module_path, excludes=None, is_file=True)
        else:
            raise ValueError(f'py_modules entry must be a directory or a .whl file; got {module_path}')
        py_modules_uris.append(module_uri)
    runtime_env['py_modules'] = py_modules_uris
    return runtime_env