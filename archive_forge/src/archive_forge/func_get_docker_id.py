import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_docker_id(self):
    if self.id:
        return self.id
    else:
        return self.docker_name()