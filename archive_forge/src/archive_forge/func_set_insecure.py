import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@staticmethod
def set_insecure(x):
    from pyspark import SparkContext
    new_val = 'True' if x else None
    SparkContext._active_spark_context.setLocalProperty('spark.databricks.ignoreTls', new_val)