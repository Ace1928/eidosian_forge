import importlib
import math
import re
from enum import Enum
def detect_credit_card(self, text):
    return re.findall(self.credit_card_regex, text)