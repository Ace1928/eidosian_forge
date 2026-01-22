import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@classmethod
def from_configs(cls, prior_configs: List[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
    """
        Instantiate a [`JukeboxConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`JukeboxConfig`]: An instance of a configuration object
        """
    prior_config_list = [config.to_dict() for config in prior_configs]
    return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)