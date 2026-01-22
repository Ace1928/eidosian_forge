from typing import Dict
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.union_operator import UnionOperator
from ray.data._internal.execution.operators.zip_operator import ZipOperator
from ray.data._internal.logical.interfaces import (
from ray.data._internal.logical.operators.all_to_all_operator import AbstractAllToAll
from ray.data._internal.logical.operators.from_operators import AbstractFrom
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.logical.operators.n_ary_operator import Union, Zip
from ray.data._internal.logical.operators.one_to_one_operator import Limit
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.planner.plan_all_to_all_op import plan_all_to_all_op
from ray.data._internal.planner.plan_from_op import plan_from_op
from ray.data._internal.planner.plan_input_data_op import plan_input_data_op
from ray.data._internal.planner.plan_limit_op import plan_limit_op
from ray.data._internal.planner.plan_read_op import plan_read_op
from ray.data._internal.planner.plan_udf_map_op import plan_udf_map_op
from ray.data._internal.planner.plan_write_op import plan_write_op
Convert logical to physical operators recursively in post-order.