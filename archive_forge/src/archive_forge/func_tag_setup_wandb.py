import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_setup_wandb():
    record_extra_usage_tag(TagKey.AIR_SETUP_WANDB_INTEGRATION_USED, '1')