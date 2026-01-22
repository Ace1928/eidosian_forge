from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def get_learner_group(framework: str, env: 'gym.Env', scaling_config: LearnerGroupScalingConfig, is_multi_agent: bool=False, eager_tracing: bool=True) -> LearnerGroup:
    """Construct a learner_group for testing.

    Args:
        framework: The framework used for training.
        env: The environment to train on.
        scaling_config: A config for the amount and types of resources to use for
            training.
        is_multi_agent: Whether to construct a multi agent rl module.
        eager_tracing: TF Specific. Whether to use tf.function for tracing
            optimizations.

    Returns:
        A learner_group.

    """
    if framework == 'tf2':
        framework_hps = FrameworkHyperparameters(eager_tracing=eager_tracing)
    else:
        framework_hps = None
    learner_spec = LearnerSpec(learner_class=get_learner_class(framework), module_spec=get_module_spec(framework=framework, env=env, is_multi_agent=is_multi_agent), learner_group_scaling_config=scaling_config, learner_hyperparameters=BaseTestingLearnerHyperparameters(), framework_hyperparameters=framework_hps)
    lg = LearnerGroup(learner_spec)
    return lg