import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
def install_generation_tagging_init():
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """
    if getattr(Module, '___needs_generation_tag_patch', True):
        init = Module.__init__

        def patched_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            GenerationTracker.tag(self)
        Module.__init__ = patched_init
        setstate = Module.__setstate__

        def patched_setstate(self, state):
            setstate(self, state)
            GenerationTracker.tag(self)
        Module.__setstate__ = patched_setstate
        Module.___needs_generation_tag_patch = False
    GenerationTracker.generation += 1