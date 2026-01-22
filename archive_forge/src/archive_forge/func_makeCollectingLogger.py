import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def makeCollectingLogger():
    """I make a logger instance that collects its logs for programmatic analysis
    -> (logger, collector)"""
    logger = logging.Logger('collector')
    handler = LogCollector()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return (logger, handler)