import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder, Model
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
@OverrideToImplementCustomLogic
def build_pi_head(self, framework: str) -> Model:
    """Builds the policy head.

        The default behavior is to build the head from the pi_head_config.
        This can be overridden to build a custom policy head as a means of configuring
        the behavior of a PPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The policy head.
        """
    action_distribution_cls = self.get_action_dist_cls(framework=framework)
    if self._model_config_dict['free_log_std']:
        _check_if_diag_gaussian(action_distribution_cls=action_distribution_cls, framework=framework)
    required_output_dim = action_distribution_cls.required_input_dim(space=self.action_space, model_config=self._model_config_dict)
    pi_head_config_class = FreeLogStdMLPHeadConfig if self._model_config_dict['free_log_std'] else MLPHeadConfig
    self.pi_head_config = pi_head_config_class(input_dims=self.latent_dims, hidden_layer_dims=self.pi_and_vf_head_hiddens, hidden_layer_activation=self.pi_and_vf_head_activation, output_layer_dim=required_output_dim, output_layer_activation='linear')
    return self.pi_head_config.build(framework=framework)