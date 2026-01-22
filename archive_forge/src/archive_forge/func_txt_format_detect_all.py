import importlib
import math
import re
from enum import Enum
def txt_format_detect_all(self, text):
    contains = self.detect_all(text)
    contains_personal_info = False
    txt = 'We believe this text contains the following personal ' + 'information:'
    for k, v in contains.items():
        if v != []:
            contains_personal_info = True
            txt += f'\n- {k.replace('_', ' ')}: {', '.join([str(x) for x in v])}'
    if not contains_personal_info:
        return ''
    return txt