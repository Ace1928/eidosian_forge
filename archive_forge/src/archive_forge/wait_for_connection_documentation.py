from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime, timedelta, timezone
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
 Test ping module, if available 