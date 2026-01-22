from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def classify_pr_data(ctx):
    if type(ctx.pr_type) is not str:
        return None
    if ctx.pr_type.startswith('GNU_PROPERTY_X86_'):
        return ('GNU_PROPERTY_X86_*', 4, 0)
    elif ctx.pr_type.startswith('GNU_PROPERTY_AARCH64_'):
        return ('GNU_PROPERTY_AARCH64_*', 4, 0)
    return (ctx.pr_type, ctx.pr_datasz, self.elfclass)