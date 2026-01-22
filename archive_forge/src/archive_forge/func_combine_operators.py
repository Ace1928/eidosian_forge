from dataclasses import dataclass
from typing import Dict, Optional, Tuple
def combine_operators(lhs: 'SelectiveBuildOperator', rhs: 'SelectiveBuildOperator') -> 'SelectiveBuildOperator':
    if str(lhs.name) != str(rhs.name):
        raise Exception(f"Expected both arguments to have the same name, but got '{str(lhs.name)}' and '{str(rhs.name)}' instead")
    return SelectiveBuildOperator(name=lhs.name, is_root_operator=lhs.is_root_operator or rhs.is_root_operator, is_used_for_training=lhs.is_used_for_training or rhs.is_used_for_training, include_all_overloads=lhs.include_all_overloads or rhs.include_all_overloads, _debug_info=merge_debug_info(lhs._debug_info, rhs._debug_info))