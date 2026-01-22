from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def roundup_padding(ctx):
    if self.elfclass == 32:
        return roundup(ctx.pr_datasz, 2) - ctx.pr_datasz
    return roundup(ctx.pr_datasz, 3) - ctx.pr_datasz