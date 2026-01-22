from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
def cleanup_arg():
    self.api.reflect_native_value(ty, native.value, self.env_manager)
    if native.cleanup is not None:
        native.cleanup()
    if self.context.enable_nrt:
        self.context.nrt.decref(self.builder, ty, native.value)