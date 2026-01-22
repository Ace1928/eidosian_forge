from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
    src = f'int {meta.orig_kernel_name}_get_num_algos(void){{\n'
    src += f'  return (int)sizeof({meta.orig_kernel_name}_kernels);\n'
    src += '}\n'
    return src