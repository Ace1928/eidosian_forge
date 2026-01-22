import contextlib
import tensorflow.compat.v2 as tf
from absl import flags
from keras.src import backend
def maybe_preemption_handler_scope(model):
    if getattr(model, '_preemption_handler', None):
        preemption_checkpoint_scope = model._preemption_handler.watch_preemption_scope()
    else:
        preemption_checkpoint_scope = contextlib.nullcontext()
    return preemption_checkpoint_scope