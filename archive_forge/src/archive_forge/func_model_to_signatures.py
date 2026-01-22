from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
def model_to_signatures(self, model: 'TFPreTrainedModel', **model_kwargs: Any) -> Dict[str, 'tf.types.experimental.ConcreteFunction']:
    """
        Creates the signatures that will be used when exporting the model to a `tf.SavedModel`.
        Each signature can be used to perform inference on the model for a given set of inputs.

        Auto-encoder models have only one signature, decoder models can have two, one for the decoder without
        caching, and one for the decoder with caching, seq2seq models can have three, and so on.
        """
    input_names = self.inputs
    output_names = self.outputs

    def forward(*args):
        if len(args) != len(input_names):
            raise ArgumentError(f'The number of inputs provided ({len(args)} do not match the number of expected inputs: {', '.join(input_names)}.')
        kwargs = dict(zip(input_names, args))
        outputs = model.call(**kwargs, **model_kwargs)
        return {key: value for key, value in outputs.items() if key in output_names}
    function = tf.function(forward, input_signature=self.inputs_specs).get_concrete_function()
    return {'model': function}