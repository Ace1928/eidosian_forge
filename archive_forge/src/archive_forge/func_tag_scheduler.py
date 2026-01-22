import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def tag_scheduler(scheduler: 'TrialScheduler'):
    from ray.tune.schedulers import TrialScheduler
    assert isinstance(scheduler, TrialScheduler)
    scheduler_name = _find_class_name(scheduler, 'ray.tune.schedulers', TUNE_SCHEDULERS)
    record_extra_usage_tag(TagKey.TUNE_SCHEDULER, scheduler_name)