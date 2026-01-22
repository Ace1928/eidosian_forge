import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
def make_dataholder(class_):
    slots = [attr.key for attr in class_._wsme_attributes]

    class DataHolder(object):
        __slots__ = slots
    DataHolder.__name__ = class_.__name__ + 'DataHolder'
    return DataHolder