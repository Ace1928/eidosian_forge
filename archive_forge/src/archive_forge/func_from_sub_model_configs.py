import os
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING
@classmethod
def from_sub_model_configs(cls, semantic_config: BarkSemanticConfig, coarse_acoustics_config: BarkCoarseConfig, fine_acoustics_config: BarkFineConfig, codec_config: PretrainedConfig, **kwargs):
    """
        Instantiate a [`BarkConfig`] (or a derived class) from bark sub-models configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        """
    return cls(semantic_config=semantic_config.to_dict(), coarse_acoustics_config=coarse_acoustics_config.to_dict(), fine_acoustics_config=fine_acoustics_config.to_dict(), codec_config=codec_config.to_dict(), **kwargs)