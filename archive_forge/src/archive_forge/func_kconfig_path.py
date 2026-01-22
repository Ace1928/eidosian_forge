from __future__ import annotations
from pathlib import Path
from lazyops.types import BaseModel, lazyproperty, validator, Field
from lazyops.configs.base import DefaultSettings, BaseSettings
from typing import List, Dict, Union, Any, Optional
@lazyproperty
def kconfig_path(self) -> Path:
    return self.kops_kconfig_path or self.kubeconfig_path