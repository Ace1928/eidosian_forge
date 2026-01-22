import copy
import inspect
import pickle
import types
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from torch import nn
import pytorch_lightning as pl
from lightning_fabric.utilities.data import AttributeDict as _AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def collect_init_args(frame: types.FrameType, path_args: List[Dict[str, Any]], inside: bool=False, classes: Tuple[Type, ...]=()) -> List[Dict[str, Any]]:
    """Recursively collects the arguments passed to the child constructors in the inheritance tree.

    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon
        classes: the classes in which to inspect the frames

    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.

    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    if not isinstance(frame.f_back, types.FrameType):
        return path_args
    local_self, local_args = _get_init_args(frame)
    if '__class__' in local_vars and (not classes or isinstance(local_self, classes)):
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside=False, classes=classes)
    return path_args