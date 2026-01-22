from collections import UserDict
from typing import Union
import numpy as np
import requests
from ..utils import (
from .audio_classification import ffmpeg_read
from .base import Pipeline, build_pipeline_init_args

        Assign labels to the audio(s) passed as inputs.

        Args:
            audios (`str`, `List[str]`, `np.array` or `List[np.array]`):
                The pipeline handles three types of inputs:
                - A string containing a http link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in numpy
            candidate_labels (`List[str]`):
                The candidate labels for this audio
            hypothesis_template (`str`, *optional*, defaults to `"This is a sound of {}"`):
                The sentence used in cunjunction with *candidate_labels* to attempt the audio classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                logits_per_audio
        Return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:
            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
        