from optparse import OptionParser
from boto.services.servicedef import ServiceDef
from boto.services.submit import Submitter
from boto.services.result import ResultProcessor
import boto
import sys, os
from boto.compat import StringIO
def do_retrieve(self):
    if not self.options.path:
        self.parser.error('No path provided')
    if not os.path.exists(self.options.path):
        self.parser.error('Invalid path (%s)' % self.options.path)
    if not self.options.batch:
        self.parser.error('batch identifier is required for retrieve command')
    s = ResultProcessor(self.options.batch, self.sd)
    s.get_results(self.options.path, get_file=not self.options.leave)