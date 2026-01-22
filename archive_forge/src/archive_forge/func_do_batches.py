from optparse import OptionParser
from boto.services.servicedef import ServiceDef
from boto.services.submit import Submitter
from boto.services.result import ResultProcessor
import boto
import sys, os
from boto.compat import StringIO
def do_batches(self):
    d = self.sd.get_obj('output_domain')
    if d:
        print('Available Batches:')
        rs = d.query("['type'='Batch']")
        for item in rs:
            print('  %s' % item.name)
    else:
        self.parser.error('No output_domain specified for service')