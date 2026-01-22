import datetime
from typing import Iterator, List, Optional, Union
import torch
from outlines.generate.generator import sequence_generator
def get_generated_token_ids(self, prompt_token_ids: torch.Tensor, token_ids: torch.Tensor) -> List[torch.Tensor]:
    """Get the tokens generated so far.

        Parameters
        ----------
        prompt_token_ids
            Tensor that contains the token ids of the sequences' prompts.
        token_ids
            The generated token ids.

        Returns
        -------
        A tensor that contains the token ids that have been generated so far.

        """
    prompt_lengths = [len(prompt) for prompt in prompt_token_ids]
    token_ids = [cur_token_ids[length:] for cur_token_ids, length in zip(token_ids, prompt_lengths)]
    return token_ids