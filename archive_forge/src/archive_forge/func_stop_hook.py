import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def stop_hook(times: List[float]) -> bool:
    if len(times) > 3:
        return common.Measurement(number_per_run=number, raw_times=times, task_spec=self._task_spec).meets_confidence(threshold=threshold)
    return False