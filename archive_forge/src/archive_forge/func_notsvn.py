import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def notsvn(path):
    return path.basename != '.svn'