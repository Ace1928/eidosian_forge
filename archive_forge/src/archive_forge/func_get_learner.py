from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def get_learner(*, framework: str, framework_hps: Optional[FrameworkHyperparameters]=None, env: 'gym.Env', learner_hps: Optional[BaseTestingLearnerHyperparameters]=None, is_multi_agent: bool=False) -> 'Learner':
    """Construct a learner for testing.

    Args:
        framework: The framework used for training.
        framework_hps: The FrameworkHyperparameters instance to pass to the
            Learner's constructor.
        env: The environment to train on.
        learner_hps: The LearnerHyperparameter instance to pass to the Learner's
            constructor.
        is_multi_agent: Whether to construct a multi agent rl module.

    Returns:
        A learner.

    """
    _cls = get_learner_class(framework)
    spec = get_module_spec(framework=framework, env=env, is_multi_agent=is_multi_agent)
    learner = _cls(module_spec=spec, learner_hyperparameters=learner_hps or BaseTestingLearnerHyperparameters(), framework_hyperparameters=framework_hps or FrameworkHyperparameters())
    learner.build()
    return learner