from typing import List, Union
from ..utils import is_torch_available
from .base import Pipeline

        Generates speech/audio from the inputs. See the [`TextToAudioPipeline`] documentation for more information.

        Args:
            text_inputs (`str` or `List[str]`):
                The text(s) to generate.
            forward_params (`dict`, *optional*):
                Parameters passed to the model generation/forward method. `forward_params` are always passed to the
                underlying model.
            generate_kwargs (`dict`, *optional*):
                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
                complete overview of generate, check the [following
                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation). `generate_kwargs` are
                only passed to the underlying model if the latter is a generative model.

        Return:
            A `dict` or a list of `dict`: The dictionaries have two keys:

            - **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.
            - **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.
        