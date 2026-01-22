import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder, Model
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
@OverrideToImplementCustomLogic
def build_actor_critic_encoder(self, framework: str) -> ActorCriticEncoder:
    """Builds the ActorCriticEncoder.

        The default behavior is to build the encoder from the encoder_config.
        This can be overridden to build a custom ActorCriticEncoder as a means of
        configuring the behavior of a PPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The ActorCriticEncoder.
        """
    return self.actor_critic_encoder_config.build(framework=framework)