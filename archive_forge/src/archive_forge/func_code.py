from collections import defaultdict
import numpy as np
import tree  # pip install dm_tree
from typing import DefaultDict, List, Optional, Set
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.util.debug import _test_some_code_for_memory_leaks, Suspect
def code():
    learner_group.update(dummy_batch)