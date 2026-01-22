from yowsup.config.v1.config import Config
from yowsup.config.transforms.dict_keyval import DictKeyValTransform
from yowsup.config.transforms.dict_json import DictJsonTransform
from yowsup.config.v1.serialize import ConfigSerialize
from yowsup.common.tools import StorageTools
import logging
import os
def config_to_str(self, config, serialize_type=TYPE_JSON):
    transform = self.get_str_transform(serialize_type)
    if transform is not None:
        return transform.transform(ConfigSerialize(config.__class__).serialize(config))
    raise ValueError('unrecognized serialize_type=%d' % serialize_type)