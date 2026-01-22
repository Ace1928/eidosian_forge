import re
import threading
def current_conf(self):
    thread_configs = local_dict().get(self._local_key)
    if thread_configs:
        return thread_configs[-1]
    elif self._process_configs:
        return self._process_configs[-1]
    else:
        return None