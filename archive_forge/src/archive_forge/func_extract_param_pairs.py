import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def extract_param_pairs(params, prefix='', keyname='', valuename=''):
    """Extract user input params from AWS style parameter-pair encoded list.

    In the AWS API list items appear as two key-value
    pairs (passed as query parameters)  with keys of the form below:

    Prefix.member.1.keyname=somekey
    Prefix.member.1.keyvalue=somevalue
    Prefix.member.2.keyname=anotherkey
    Prefix.member.2.keyvalue=somevalue

    We reformat this into a dict here to match the heat
    engine API expected format.
    """
    plist = extract_param_list(params, prefix)
    kvs = [(p[keyname], p[valuename]) for p in plist if keyname in p and valuename in p]
    return dict(kvs)