from typing import Optional, Type, TYPE_CHECKING, Union
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.from_config import NotProvided
def learner(self, *, learner_class: Optional[Type['Learner']]=NotProvided, learner_hyperparameters: Optional[LearnerHyperparameters]=NotProvided) -> 'LearnerGroupConfig':
    if learner_class is not NotProvided:
        self.learner_class = learner_class
    if learner_hyperparameters is not NotProvided:
        self.learner_hyperparameters = learner_hyperparameters
    return self