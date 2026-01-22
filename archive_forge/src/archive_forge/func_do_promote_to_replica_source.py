import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
def do_promote_to_replica_source(cs, args):
    """Promotes a replica to be the new replica source of its set."""
    instance = _find_instance(cs, args.instance)
    cs.instances.promote_to_replica_source(instance)