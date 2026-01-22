import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
def total_resources(self) -> Dict[str, float]:
    return {r.resource_name: r.total for r in self.cluster_resource_usage}