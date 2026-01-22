import copy
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
@classmethod
def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
    """Instantiate a [`TvpConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`TvpConfig`]: An instance of a configuration object
        """
    return cls(backbone_config=backbone_config, **kwargs)