import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def midl(name, default=None):
    return self.ConvertVSMacros(midl_get(name, default=default), config=config)