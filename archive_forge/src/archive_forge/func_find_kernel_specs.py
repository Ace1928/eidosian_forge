from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def find_kernel_specs(self) -> dict[str, str]:
    """Returns a dict mapping kernel names to resource directories."""
    d = {}
    for kernel_dir in self.kernel_dirs:
        kernels = _list_kernels_in(kernel_dir)
        for kname, spec in kernels.items():
            if kname not in d:
                self.log.debug('Found kernel %s in %s', kname, kernel_dir)
                d[kname] = spec
    if self.ensure_native_kernel and NATIVE_KERNEL_NAME not in d:
        try:
            from ipykernel.kernelspec import RESOURCES
            self.log.debug('Native kernel (%s) available from %s', NATIVE_KERNEL_NAME, RESOURCES)
            d[NATIVE_KERNEL_NAME] = RESOURCES
        except ImportError:
            self.log.warning('Native kernel (%s) is not available', NATIVE_KERNEL_NAME)
    if self.allowed_kernelspecs:
        d = {name: spec for name, spec in d.items() if name in self.allowed_kernelspecs}
    return d