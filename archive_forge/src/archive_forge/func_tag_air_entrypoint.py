import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_air_entrypoint(entrypoint: AirEntrypoint) -> None:
    """Records the entrypoint to an AIR training run."""
    assert entrypoint in AirEntrypoint
    record_extra_usage_tag(TagKey.AIR_ENTRYPOINT, entrypoint.value)