import importlib
import pkgutil
from types import ModuleType
from typing import List, Optional
import numpy as np
from onnx import ONNX_ML
def import_recursive(package: ModuleType) -> None:
    """Takes a package and imports all modules underneath it."""
    pkg_dir: Optional[List[str]] = None
    pkg_dir = package.__path__
    module_location = package.__name__
    for _module_loader, name, ispkg in pkgutil.iter_modules(pkg_dir):
        module_name = f'{module_location}.{name}'
        if not ONNX_ML and module_name.startswith('onnx.backend.test.case.node.ai_onnx_ml'):
            continue
        module = importlib.import_module(module_name)
        if ispkg:
            import_recursive(module)