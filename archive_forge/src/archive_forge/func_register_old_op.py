import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional, List
import torch
from torch._export import export
from torch._export.pass_base import _ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep
import re
def register_old_op(name: str, schema: str, impl_str: str):
    """Registers an old version operator using impl_name as old op name."""
    lib.define(schema)
    try:
        exec(impl_str)
    except Exception as e:
        raise RuntimeError(f'Invalid upgrader string: {impl_str}') from e
    impl_lib.impl(name, locals()[name], 'CompositeImplicitAutograd')