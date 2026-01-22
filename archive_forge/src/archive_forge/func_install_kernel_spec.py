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
def install_kernel_spec(self, source_dir: str, kernel_name: str | None=None, user: bool=False, replace: bool | None=None, prefix: str | None=None) -> str:
    """Install a kernel spec by copying its directory.

        If ``kernel_name`` is not given, the basename of ``source_dir`` will
        be used.

        If ``user`` is False, it will attempt to install into the systemwide
        kernel registry. If the process does not have appropriate permissions,
        an :exc:`OSError` will be raised.

        If ``prefix`` is given, the kernelspec will be installed to
        PREFIX/share/jupyter/kernels/KERNEL_NAME. This can be sys.prefix
        for installation inside virtual or conda envs.
        """
    source_dir = source_dir.rstrip('/\\')
    if not kernel_name:
        kernel_name = os.path.basename(source_dir)
    kernel_name = kernel_name.lower()
    if not _is_valid_kernel_name(kernel_name):
        msg = f'Invalid kernel name {kernel_name!r}.  {_kernel_name_description}'
        raise ValueError(msg)
    if user and prefix:
        msg = "Can't specify both user and prefix. Please choose one or the other."
        raise ValueError(msg)
    if replace is not None:
        warnings.warn('replace is ignored. Installing a kernelspec always replaces an existing installation', DeprecationWarning, stacklevel=2)
    destination = self._get_destination_dir(kernel_name, user=user, prefix=prefix)
    self.log.debug('Installing kernelspec in %s', destination)
    kernel_dir = os.path.dirname(destination)
    if kernel_dir not in self.kernel_dirs:
        self.log.warning('Installing to %s, which is not in %s. The kernelspec may not be found.', kernel_dir, self.kernel_dirs)
    if os.path.isdir(destination):
        self.log.info('Removing existing kernelspec in %s', destination)
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    self.log.info('Installed kernelspec %s in %s', kernel_name, destination)
    return destination