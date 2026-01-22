import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
def setup_image_features(self, data_path):
    """
        Load text and image data.

        The image features all live in dicts by default in <data_path>/
        image_features/ but get_image_features_path() above can be overriden by
        subclass to put them elsewhere.

        In the (very odd) case that the resnet or resnext dicts (models
        buildable using ImageLoader) are not found, we build them.
        """
    if self.image_mode in ['raw', 'ascii']:
        self.image_features_dict = None
        self.image_loader = ImageLoader(self.opt)
        return
    image_mode_features_dict_path = self.get_image_features_path(self.task, self.image_mode, self.datatype)
    if os.path.isfile(image_mode_features_dict_path):
        logging.info(f'Loading existing image features dict for model: {self.image_mode} at: {image_mode_features_dict_path}')
        self.image_features_dict = torch.load(image_mode_features_dict_path, map_location='cpu')
    else:
        logging.warn('No existing image features, attempting to build.')
        if self.is_image_mode_buildable(self.image_mode):
            image_loader_opt = self.opt.copy()
            image_loader_opt['image_mode'] = self.image_mode if self.include_image else 'no_image_model'
            image_loader_opt['image_size'] = 256
            image_loader_opt['image_cropsize'] = 224
            self.image_loader = ImageLoader(image_loader_opt)
            self.image_features_dict = self._build_image_features_dict(self.data_path, self.datatype, image_mode_features_dict_path)
        else:
            raise RuntimeError('Image model: %s is not buildable by ImageLoader but doesnot already exist on disk as an image features dict forthis dataset.' % self.image_mode)