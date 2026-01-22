import os
import subprocess
import io
import time
import threading
import Pyro4
def launch_im():
    return subprocess.Popen('python3 scripts/launch_instance_manager.py --seeding_type=3 --seeds=1,1,1,1;2,2,2,2'.split(' '), shell=False)