import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def filter_domain(domain):
    lpart, found_separator, rpart = domain.partition(']')
    res = lpart.lstrip('[')
    if not found_separator:
        lpart, found_separator, rpart = domain.rpartition(':')
        res = lpart if found_separator else rpart
    return res.lower().strip().rstrip('.')