import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def get_unique_tags(field_to_obs):
    """Returns a dictionary of tags that a user could query over.

    Args:
      field_to_obs: Dict that maps string field to `Observation` list.

    Returns:
      A dict that maps keys in `TAG_FIELDS` to a list of string tags present in
      the event files. If the dict does not have any observations of the type,
      maps to an empty list so that we can render this to console.
    """
    return {field: sorted(set([x.get('tag', '') for x in observations])) for field, observations in field_to_obs.items() if field in TAG_FIELDS}