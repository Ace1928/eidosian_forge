from yowsup.config.manager import ConfigManager
from yowsup.config.v1.config import Config
from yowsup.axolotl.manager import AxolotlManager
from yowsup.axolotl.factory import AxolotlManagerFactory
import logging
@property
def axolotl_manager(self):
    if self._axolotl_manager is None:
        self._axolotl_manager = self._load_axolotl_manager()
    return self._axolotl_manager