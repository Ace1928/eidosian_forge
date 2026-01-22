import pathlib
from typing import Optional, Union
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
@lazyproperty
def in_cluster(self) -> bool:
    return self.kubernetes_service_host is not None and self.kubernetes_service_port is not None or self.in_k8s