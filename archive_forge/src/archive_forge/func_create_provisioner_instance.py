import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def create_provisioner_instance(self, kernel_id: str, kernel_spec: Any, parent: Any) -> KernelProvisionerBase:
    """
        Reads the associated ``kernel_spec`` to see if it has a `kernel_provisioner` stanza.
        If one exists, it instantiates an instance.  If a kernel provisioner is not
        specified in the kernel specification, a default provisioner stanza is fabricated
        and instantiated corresponding to the current value of ``default_provisioner_name`` trait.
        The instantiated instance is returned.

        If the provisioner is found to not exist (not registered via entry_points),
        `ModuleNotFoundError` is raised.
        """
    provisioner_cfg = self._get_provisioner_config(kernel_spec)
    provisioner_name = str(provisioner_cfg.get('provisioner_name'))
    if not self._check_availability(provisioner_name):
        msg = f"Kernel provisioner '{provisioner_name}' has not been registered."
        raise ModuleNotFoundError(msg)
    self.log.debug(f"Instantiating kernel '{kernel_spec.display_name}' with kernel provisioner: {provisioner_name}")
    provisioner_class = self.provisioners[provisioner_name].load()
    provisioner_config = provisioner_cfg.get('config')
    provisioner: KernelProvisionerBase = provisioner_class(kernel_id=kernel_id, kernel_spec=kernel_spec, parent=parent, **provisioner_config)
    return provisioner