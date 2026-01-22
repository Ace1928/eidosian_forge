import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def get_plan_as_string(self, classname: str) -> str:
    """Create a cosmetic string representation of this execution plan.

        Returns:
            The string representation of this execution plan.
        """
    plan_str = ''
    num_stages = 0
    dataset_blocks = None
    if self._stages_after_snapshot:
        for stage in self._stages_after_snapshot[::-1]:
            stage_str = stage.name.split('(')
            stage_str[0] = capitalize(stage_str[0])
            stage_name = '('.join(stage_str)
            if num_stages == 0:
                plan_str += f'{stage_name}\n'
            else:
                trailing_space = ' ' * ((num_stages - 1) * 3)
                plan_str += f'{trailing_space}+- {stage_name}\n'
            num_stages += 1
        if self._snapshot_blocks is not None:
            schema = self._get_unified_blocks_schema(self._snapshot_blocks, fetch_if_missing=False)
            dataset_blocks = self._snapshot_blocks
        else:
            assert self._in_blocks is not None
            schema = self._get_unified_blocks_schema(self._in_blocks, fetch_if_missing=False)
            dataset_blocks = self._in_blocks
    else:
        schema = self.schema(fetch_if_missing=False)
        dataset_blocks = self._snapshot_blocks
    if schema is None:
        schema_str = 'Unknown schema'
    elif isinstance(schema, type):
        schema_str = str(schema)
    else:
        schema_str = []
        for n, t in zip(schema.names, schema.types):
            if hasattr(t, '__name__'):
                t = t.__name__
            schema_str.append(f'{n}: {t}')
        schema_str = ', '.join(schema_str)
        schema_str = '{' + schema_str + '}'
    count = self._get_num_rows_from_blocks_metadata(dataset_blocks)
    if count is None:
        count = '?'
    if dataset_blocks is None:
        num_blocks = '?'
    else:
        num_blocks = dataset_blocks.estimated_num_blocks()
    name_str = 'name={}, '.format(self._dataset_name) if self._dataset_name is not None else ''
    dataset_str = '{}({}num_blocks={}, num_rows={}, schema={})'.format(classname, name_str, num_blocks, count, schema_str)
    SCHEMA_LINE_CHAR_LIMIT = 80
    MIN_FIELD_LENGTH = 10
    INDENT_STR = ' ' * 3
    trailing_space = ' ' * (max(num_stages, 0) * 3)
    if len(dataset_str) > SCHEMA_LINE_CHAR_LIMIT:
        schema_str_on_new_line = f'{trailing_space}{INDENT_STR}schema={schema_str}'
        if len(schema_str_on_new_line) > SCHEMA_LINE_CHAR_LIMIT:
            schema_str = []
            for n, t in zip(schema.names, schema.types):
                if hasattr(t, '__name__'):
                    t = t.__name__
                col_str = f'{trailing_space}{INDENT_STR * 2}{n}: {t}'
                if len(col_str) > SCHEMA_LINE_CHAR_LIMIT:
                    shortened_suffix = f'...: {str(t)}'
                    chars_left_for_col_name = max(SCHEMA_LINE_CHAR_LIMIT - len(shortened_suffix), MIN_FIELD_LENGTH)
                    col_str = f'{col_str[:chars_left_for_col_name]}{shortened_suffix}'
                schema_str.append(col_str)
            schema_str = ',\n'.join(schema_str)
            schema_str = '{\n' + schema_str + f'\n{trailing_space}{INDENT_STR}' + '}'
        name_str = f'\n{trailing_space}{INDENT_STR}name={self._dataset_name},' if self._dataset_name is not None else ''
        dataset_str = f'{classname}({name_str}\n{trailing_space}{INDENT_STR}num_blocks={num_blocks},\n{trailing_space}{INDENT_STR}num_rows={count},\n{trailing_space}{INDENT_STR}schema={schema_str}\n{trailing_space})'
    if num_stages == 0:
        plan_str = dataset_str
    else:
        trailing_space = ' ' * ((num_stages - 1) * 3)
        plan_str += f'{trailing_space}+- {dataset_str}'
    return plan_str