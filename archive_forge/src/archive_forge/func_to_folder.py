import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
@compatibility(is_backward_compatible=False)
def to_folder(self, folder: Union[str, os.PathLike], module_name: str='FxModule'):
    """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
    folder = Path(folder)
    Path(folder).mkdir(exist_ok=True)
    torch.save(self.state_dict(), folder / 'state_dict.pt')
    tab = ' ' * 4
    custom_builtins = '\n'.join([v.import_str for v in _custom_builtins.values()])
    model_str = f'\nimport torch\n{custom_builtins}\n\nfrom torch.nn import *\nclass {module_name}(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n'

    def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
        safe_reprs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        if type(module) in safe_reprs:
            return f'{module.__repr__()}'
        else:
            return None
    blobified_modules = []
    for module_name, module in self.named_children():
        module_str = _gen_model_repr(module_name, module)
        if module_str is None:
            module_file = folder / f'{module_name}.pt'
            torch.save(module, module_file)
            blobified_modules.append(module_name)
            module_repr = module.__repr__().replace('\r', ' ').replace('\n', ' ')
            module_str = f"torch.load(r'{module_file}') # {module_repr}"
        model_str += f'{tab * 2}self.{module_name} = {module_str}\n'
    for buffer_name, buffer in self._buffers.items():
        if buffer is None:
            continue
        model_str += f"{tab * 2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"
    for param_name, param in self._parameters.items():
        if param is None:
            continue
        model_str += f'{tab * 2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n'
    model_str += f"{tab * 2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
    model_str += f'{_addindent(self.code, 4)}\n'
    module_file = folder / 'module.py'
    module_file.write_text(model_str)
    init_file = folder / '__init__.py'
    init_file.write_text('from .module import *')
    if len(blobified_modules) > 0:
        warnings.warn(f'Was not able to save the following children modules as reprs -saved as pickled files instead: {blobified_modules}')