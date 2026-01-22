from yowsup.config.v1.config import Config
from yowsup.config.transforms.dict_keyval import DictKeyValTransform
from yowsup.config.transforms.dict_json import DictJsonTransform
from yowsup.config.v1.serialize import ConfigSerialize
from yowsup.common.tools import StorageTools
import logging
import os
def guess_type(self, config_path):
    dissected = os.path.splitext(config_path)
    if len(dissected) > 1:
        ext = dissected[1][1:].lower()
        config_type = self.MAP_EXT[ext] if ext in self.MAP_EXT else None
    else:
        config_type = None
    if config_type is not None:
        return config_type
    else:
        logger.debug('Trying auto detect config type by parsing')
        with open(config_path, 'r') as f:
            data = f.read()
        for config_type, transform in self.TYPES.items():
            config_type_str = self.TYPE_NAMES[config_type]
            try:
                logger.debug('Trying to parse as %s' % config_type_str)
                if transform().reverse(data):
                    logger.debug('Successfully detected %s as config type for %s' % (config_type_str, config_path))
                    return config_type
            except Exception as ex:
                logger.debug('%s was not parseable as %s, reason: %s' % (config_path, config_type_str, ex))