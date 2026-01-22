import copy
from botocore.utils import merge_dicts
Recursively replace $ref keys.

    To cut down on duplication, common definitions can be declared
    (and passed in via the ``definitions`` attribute) and then
    references as {"$ref": "name"}, when this happens the reference
    dict is placed with the value from the ``definition`` dict.

    This is recursively done.

    