import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def relative_batch_tokens_ids_to_midi(self, tokens: np.ndarray, beatstep: np.ndarray, beat_offset_idx: int=0, bars_per_batch: int=2, cutoff_time_idx: int=12):
    """
        Converts tokens to Midi. This method calls `relative_batch_tokens_ids_to_notes` method to convert batch tokens
        to notes then uses `notes_to_midi` method to convert them to Midi.

        Args:
            tokens (`numpy.ndarray`):
                Denotes tokens which alongside beatstep will be converted to Midi.
            beatstep (`np.ndarray`):
                We get beatstep from feature extractor which is also used to get Midi.
            beat_offset_idx (`int`, *optional*, defaults to 0):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`, *optional*, defaults to 2):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`, *optional*, defaults to 12):
                Denotes the cutoff time index for each note in generated Midi.
        """
    beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
    notes = self.relative_batch_tokens_ids_to_notes(tokens=tokens, beat_offset_idx=beat_offset_idx, bars_per_batch=bars_per_batch, cutoff_time_idx=cutoff_time_idx)
    midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
    return midi