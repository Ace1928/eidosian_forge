from yowsup.config.manager import ConfigManager
from yowsup.config.v1.config import Config
from yowsup.axolotl.manager import AxolotlManager
from yowsup.axolotl.factory import AxolotlManagerFactory
import logging

        :param profile_name:  profile name
        :param config: A supplied config will disable loading configs using the Config manager and provide that config
        instead
        