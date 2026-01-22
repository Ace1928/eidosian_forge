import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
@classmethod
def create_optparser(cls, load_file):
    oparser = optparse.OptionParser(usage='%prog [options] <cmd> <action> <args>', version='1.0', conflict_handler='resolve')
    if load_file:
        file = cls.load_from_file()
    else:
        file = cls.default()

    def add_option(*args, **kwargs):
        if len(args) == 1:
            name = args[0]
        else:
            name = args[1]
        kwargs['default'] = getattr(file, name, cls.DEFAULT_VALUES[name])
        oparser.add_option('--%s' % name, **kwargs)
    add_option('verbose', action='store_true', help='Show equivalent curl statement along with actual HTTP communication.')
    add_option('debug', action='store_true', help='Show the stack trace on errors.')
    add_option('auth_url', help='Auth API endpoint URL with port and version. Default: http://localhost:5000/v2.0')
    add_option('username', help='Login username.')
    add_option('apikey', help='API key.')
    add_option('tenant_id', help='Tenant Id associated with the account.')
    add_option('auth_type', help="Auth type to support different auth environments, Supported value are 'keystone'.")
    add_option('service_type', help='Service type is a name associated for the catalog.')
    add_option('service_name', help='Service name as provided in the service catalog.')
    add_option('service_url', help="Service endpoint to use if the catalog doesn't have one.")
    add_option('region', help='Region the service is located in.')
    add_option('insecure', action='store_true', help='Run in insecure mode for https endpoints.')
    add_option('token', help='Token from a prior login.')
    oparser.add_option('--json', action='store_false', dest='xml', help='Changes format to JSON.')
    oparser.add_option('--secure', action='store_false', dest='insecure', help='Run in insecure mode for https endpoints.')
    oparser.add_option('--terse', action='store_false', dest='verbose', help='Toggles verbose mode off.')
    oparser.add_option('--hide-debug', action='store_false', dest='debug', help='Toggles debug mode off.')
    return oparser