import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def compressed_steps(steps):
    return {'num_steps': len(set(steps)), 'min_step': min(steps), 'max_step': max(steps), 'last_step': steps[-1], 'first_step': steps[0], 'outoforder_steps': get_out_of_order(steps)}