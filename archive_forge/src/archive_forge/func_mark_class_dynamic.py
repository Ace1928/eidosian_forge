import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
@staticmethod
def mark_class_dynamic(cls):
    assert issubclass(cls, torch.nn.Module)
    GenerationTracker.dynamic_classes[cls] = True