import itertools
import torch
from torch.autograd.profiler_legacy import profile
from typing import List
from . import (

        Turn off server-side process-global profiling.
        Aggregate all profiling events recorded by RPC threads.

        These attributes are assigned on exiting context.

        Attributes:
            function_events (torch.autograd.profiler.EventList).  It's a list that has helper
            methods, like 1) show record items in a pretty-print table.
            2) do averaging by grouping on keys. 3) and more.

            process_global_function_events (List[torch.autograd.profiler.FunctionEvent]).
            It's a list of ``FunctionEvent`` elements. Every element is a profiling result
            of an RPC request handling within the profiling range.
        