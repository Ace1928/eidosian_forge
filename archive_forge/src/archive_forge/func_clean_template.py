import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def clean_template(self, template):
    """Remove preprocsection or outputsection"""
    if not self.dopreproc and self.options.get('codeonly'):
        r = re.compile('<<startcodeonlysection>>(.*?)<<endcodeonlysection>>', re.DOTALL | re.MULTILINE)
        m = r.search(template)
        if m:
            return m.group(1).strip()
    if not self.dopreproc and self.options.get('figonly'):
        r = re.compile('<<start_figonlysection>>(.*?)<<end_figonlysection>>', re.DOTALL | re.MULTILINE)
        m = r.search(template)
        if m:
            return m.group(1)
        r = re.compile('<<startfigonlysection>>(.*?)<<endfigonlysection>>', re.DOTALL | re.MULTILINE)
        m = r.search(template)
        if m:
            return m.group(1)
    if self.dopreproc:
        r = re.compile('<<startoutputsection>>.*?<<endoutputsection>>', re.DOTALL | re.MULTILINE)
    else:
        r = re.compile('<<startpreprocsection>>.*?<<endpreprocsection>>', re.DOTALL | re.MULTILINE)
    r2 = re.compile('<<start_figonlysection>>.*?<<end_figonlysection>>', re.DOTALL | re.MULTILINE)
    tmp = r2.sub('', template)
    r2 = re.compile('<<startcodeonlysection>>.*?<<endcodeonlysection>>', re.DOTALL | re.MULTILINE)
    tmp = r2.sub('', tmp)
    return r.sub('', tmp)