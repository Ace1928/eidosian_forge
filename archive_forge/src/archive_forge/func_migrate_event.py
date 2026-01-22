import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util
def migrate_event(event):
    if not event.HasField('summary'):
        return event
    old_values = event.summary.value
    new_values = [migrate_value(value) for value in old_values]
    if len(old_values) == len(new_values) and all((x is y for x, y in zip(old_values, new_values))):
        return event
    result = event_pb2.Event()
    result.CopyFrom(event)
    del result.summary.value[:]
    result.summary.value.extend(new_values)
    return result