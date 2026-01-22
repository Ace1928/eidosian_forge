import re
import threading
def push_process_config(self, conf):
    """
        Like push_thread_config, but applies the configuration to
        the entire process.
        """
    self._process_configs.append(conf)