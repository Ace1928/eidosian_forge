import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
def do_eject_replica_source(cs, args):
    """Ejects a replica source from its set."""
    instance = _find_instance(cs, args.instance)
    cs.instances.eject_replica_source(instance)