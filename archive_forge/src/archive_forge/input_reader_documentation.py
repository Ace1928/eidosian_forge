from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import threading
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.framework import try_import_tf
from typing import Dict, List
from ray.rllib.utils.typing import TensorType, SampleBatchType
Returns TensorFlow queue ops for reading inputs from this reader.

        The main use of these ops is for integration into custom model losses.
        For example, you can use tf_input_ops() to read from files of external
        experiences to add an imitation learning loss to your model.

        This method creates a queue runner thread that will call next() on this
        reader repeatedly to feed the TensorFlow queue.

        Args:
            queue_size: Max elements to allow in the TF queue.

        .. testcode::
            :skipif: True

            from ray.rllib.models.modelv2 import ModelV2
            from ray.rllib.offline.json_reader import JsonReader
            imitation_loss = ...
            class MyModel(ModelV2):
                def custom_loss(self, policy_loss, loss_inputs):
                    reader = JsonReader(...)
                    input_ops = reader.tf_input_ops()
                    logits, _ = self._build_layers_v2(
                        {"obs": input_ops["obs"]},
                        self.num_outputs, self.options)
                    il_loss = imitation_loss(logits, input_ops["action"])
                    return policy_loss + il_loss

        You can find a runnable version of this in examples/custom_loss.py.

        Returns:
            Dict of Tensors, one for each column of the read SampleBatch.
        