from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def get_final_hypothesis(self) -> List[CTCHypothesis]:
    """Get the final hypothesis

        Returns:
            List[CTCHypothesis]:
                List of sorted best hypotheses.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.
        """
    results = self.decoder.get_all_final_hypothesis()
    return self._to_hypo(results[:self.nbest])