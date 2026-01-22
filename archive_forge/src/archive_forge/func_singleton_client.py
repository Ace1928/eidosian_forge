import ipaddress
import time
from datetime import datetime
from enum import Enum
def singleton_client(cls):
    """
    A singleton decorator helps us to make sure there is only one instance
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = (cls(*args, **kwargs), time.time())
        else:
            instance, last_checked_time = instances[cls]
            current_time = time.time()
            if current_time - last_checked_time > Constants.ENSURE_CONNECTION_PERIOD:
                instance.ensure_connect()
                instances[cls] = (instance, current_time)
        return instances[cls][0]
    return get_instance