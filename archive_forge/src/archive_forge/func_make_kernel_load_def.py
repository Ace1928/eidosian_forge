from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_kernel_load_def(names: str, meta: KernelLinkerMeta) -> str:
    src = ''
    for mode in ['load', 'unload']:
        src += f'void {mode}_{meta.orig_kernel_name}(void){{\n'
        for name in names:
            src += f'  {mode}_{name}();\n'
        src += '}\n\n'
    return src