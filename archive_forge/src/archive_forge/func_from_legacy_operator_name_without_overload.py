from dataclasses import dataclass
from typing import Dict, Optional, Tuple
@staticmethod
def from_legacy_operator_name_without_overload(name: str) -> 'SelectiveBuildOperator':
    return SelectiveBuildOperator(name=name, is_root_operator=True, is_used_for_training=True, include_all_overloads=True, _debug_info=None)