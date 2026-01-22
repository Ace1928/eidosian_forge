from typing import Dict, List, Sequence, Tuple
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager
from .gen_inplace_or_view_type import VIEW_FUNCTIONS
def get_infos_with_derivatives_list(differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]]) -> List[DifferentiabilityInfo]:
    diff_info_list = [info for diffinfo_dict in differentiability_infos.values() for info in diffinfo_dict.values()]
    return list(filter(lambda info: info.args_with_derivatives, diff_info_list))