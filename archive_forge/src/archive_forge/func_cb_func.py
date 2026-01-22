import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def cb_func(tag, name, desc, file):
    tag = tag.decode()
    name = name.decode()
    desc = desc.decode()
    file = file.decode()
    cb_result.append((tag, name, desc, file))