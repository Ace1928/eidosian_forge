import collections
import os
import re
import sys
import functools
import itertools
def win32_is_iot():
    return win32_edition() in ('IoTUAP', 'NanoServer', 'WindowsCoreHeadless', 'IoTEdgeOS')