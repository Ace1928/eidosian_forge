import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
@classmethod
def register_datapipe_as_function(cls, function_name, cls_to_register):
    if function_name in cls.functions:
        raise Exception(f'Unable to add DataPipe function name {function_name} as it is already taken')

    def class_function(cls, source_dp, *args, **kwargs):
        result_pipe = cls(source_dp, *args, **kwargs)
        return result_pipe
    function = functools.partial(class_function, cls_to_register)
    functools.update_wrapper(wrapper=function, wrapped=cls_to_register, assigned=('__doc__',))
    cls.functions[function_name] = function