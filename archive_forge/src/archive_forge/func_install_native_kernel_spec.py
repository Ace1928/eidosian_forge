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
def install_native_kernel_spec(self, user: bool=False) -> None:
    """DEPRECATED: Use ipykernel.kernelspec.install"""
    warnings.warn('install_native_kernel_spec is deprecated. Use ipykernel.kernelspec import install.', stacklevel=2)
    from ipykernel.kernelspec import install
    install(self, user=user)