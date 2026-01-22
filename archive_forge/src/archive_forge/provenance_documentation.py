from collections import OrderedDict
from copy import deepcopy
from pickle import dumps
import os
import getpass
import platform
from uuid import uuid1
import simplejson as json
import numpy as np
import prov.model as pm
from .. import get_info, logging, __version__
from .filemanip import md5, hashlib, hash_infile

    Encodes a python value for prov
    