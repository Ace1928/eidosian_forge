from typing import Dict
import numpy as np
import torch as th
import torch.nn as nn
from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel
def reorder_decoder_incremental_state(self, incremental_state: Dict[int, dict], inds: th.Tensor) -> Dict[int, dict]:
    """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
    return {idx: layer.reorder_incremental_state(incremental_state[idx], inds) for idx, layer in enumerate(self.decoder.transformer.layers)}