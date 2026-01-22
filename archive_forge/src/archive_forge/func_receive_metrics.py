from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm
def receive_metrics(self, metrics_dict):
    """
        Receive the metrics from validation.

        Unfreeze text encoder weights after a certain number of rounds without improvement.

        :param metrics_dict:
            the metrics dictionary
        """
    if 'tasks' in metrics_dict:
        metrics_dict = metrics_dict['tasks']['personality_captions']
    if self.freeze_patience != -1 and self.is_frozen:
        m = metrics_dict['hits@1/100']
        if m > self.freeze_best_metric:
            self.freeze_impatience = 0
            self.freeze_best_metric = m
            print('performance not good enough to unfreeze the model.')
        else:
            self.freeze_impatience += 1
            print('Growing impatience for unfreezing')
            if self.freeze_impatience >= self.freeze_patience:
                self.is_frozen = False
                print('Reached impatience for fine tuning. Reloading the best model so far.')
                self._build_model(self.model_file)
                if self.use_cuda:
                    self.model = self.model.cuda()
                print('Unfreezing.')
                self.model.unfreeze_text_encoder()
                print('Done')