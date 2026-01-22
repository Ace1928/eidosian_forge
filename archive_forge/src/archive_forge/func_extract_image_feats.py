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
def extract_image_feats(self, obs):
    """
        Extract image features from the observations.

        :param obs:
            list of observations

        :return:
            list of image features
        """
    tmp_image_feats = [v.get('image') for v in obs]
    for i, im in enumerate(tmp_image_feats):
        try:
            if len(im.size()) == 4:
                tmp_image_feats[i] = im[0, :, 0, 0]
        except TypeError:
            tmp_image_feats[i] = self.blank_image_features
    image_feats = []
    for img in tmp_image_feats:
        image_feats.append(img.detach())
    return image_feats