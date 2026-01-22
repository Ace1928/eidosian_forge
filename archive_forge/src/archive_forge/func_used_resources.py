import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
def used_resources(self) -> Dict[str, float]:
    if self.resource_usage is None:
        return {}
    return {r.resource_name: r.used for r in self.resource_usage.usage}