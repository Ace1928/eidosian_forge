from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
@staticmethod
def from_yaml_dict(data: Dict[str, object]) -> 'SelectiveBuilder':
    valid_top_level_keys = {'include_all_non_op_selectives', 'include_all_operators', 'debug_info', 'operators', 'kernel_metadata', 'et_kernel_metadata', 'custom_classes', 'build_features'}
    top_level_keys = set(data.keys())
    if len(top_level_keys - valid_top_level_keys) > 0:
        raise Exception('Got unexpected top level keys: {}'.format(','.join(top_level_keys - valid_top_level_keys)))
    include_all_operators = data.get('include_all_operators', False)
    assert isinstance(include_all_operators, bool)
    debug_info = None
    if 'debug_info' in data:
        di_list = data['debug_info']
        assert isinstance(di_list, list)
        debug_info = tuple((str(x) for x in di_list))
    operators = {}
    operators_dict = data.get('operators', {})
    assert isinstance(operators_dict, dict)
    for k, v in operators_dict.items():
        operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)
    kernel_metadata = {}
    kernel_metadata_dict = data.get('kernel_metadata', {})
    assert isinstance(kernel_metadata_dict, dict)
    for k, v in kernel_metadata_dict.items():
        kernel_metadata[str(k)] = [str(dtype) for dtype in v]
    et_kernel_metadata = data.get('et_kernel_metadata', {})
    assert isinstance(et_kernel_metadata, dict)
    custom_classes = data.get('custom_classes', [])
    assert isinstance(custom_classes, Iterable)
    custom_classes = set(custom_classes)
    build_features = data.get('build_features', [])
    assert isinstance(build_features, Iterable)
    build_features = set(build_features)
    include_all_non_op_selectives = data.get('include_all_non_op_selectives', False)
    assert isinstance(include_all_non_op_selectives, bool)
    return SelectiveBuilder(include_all_operators, debug_info, operators, kernel_metadata, et_kernel_metadata, custom_classes, build_features, include_all_non_op_selectives)