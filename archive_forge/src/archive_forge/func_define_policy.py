import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def define_policy(self, policy):
    self.policies[policy['name']] = policy