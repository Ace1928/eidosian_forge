import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
def print_it(self):
    if not self.printed:
        print(self.line)
        self.printed = True