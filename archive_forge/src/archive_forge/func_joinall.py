import logging
import os
from os_ken.lib import ip
def joinall(threads):
    for t in threads:
        try:
            t.wait()
        except TaskExit:
            pass