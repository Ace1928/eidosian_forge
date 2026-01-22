from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
def run_playbook(self, playbook: str, variables: dict[str, str]) -> None:
    """Run the specified playbook using the current inventory."""
    self.create_inventory()
    run_playbook(self.args, self.inventory_path, playbook, capture=False, variables=variables)