from __future__ import annotations
from pathlib import Path
from lazyops.types import BaseModel, lazyproperty, validator, Field
from lazyops.configs.base import DefaultSettings, BaseSettings
from typing import List, Dict, Union, Any, Optional
@lazyproperty
def kubeconfig_path(self):
    if not self.kubeconfig:
        return None
    path = Path(self.kubeconfig)
    return path if path.exists() else None