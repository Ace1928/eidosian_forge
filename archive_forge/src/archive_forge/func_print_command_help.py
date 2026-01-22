from optparse import OptionParser
from boto.services.servicedef import ServiceDef
from boto.services.submit import Submitter
from boto.services.result import ResultProcessor
import boto
import sys, os
from boto.compat import StringIO
def print_command_help(self):
    print('\nCommands:')
    for key in self.Commands.keys():
        print('  %s\t\t%s' % (key, self.Commands[key]))