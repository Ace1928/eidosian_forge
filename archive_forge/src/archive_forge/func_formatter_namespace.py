import argparse
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
@property
def formatter_namespace(self):
    return 'vitrageclient.formatter.show'