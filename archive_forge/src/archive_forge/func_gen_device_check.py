import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
@staticmethod
def gen_device_check(type: DeviceCheckType, args: List[Argument], method_name: str) -> str:
    if type == DeviceCheckType.NoCheck:
        return '  // No device check\n'
    device_check = 'c10::optional<Device> common_device = nullopt;\n'
    device_check += '(void)common_device; // Suppress unused variable warning\n'
    for arg in args:
        if arg.type.is_tensor_like():
            device_check += f'\n  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");'
    return device_check