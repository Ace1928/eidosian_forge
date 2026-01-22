import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('Id or Name of the instance.'))
@utils.arg('log_name', metavar='<log_name>', help=_('Name of log to publish.'))
@utils.arg('--lines', metavar='<lines>', default=50, type=int, help=_('Publish latest entries from guest before display.'))
@utils.service_type('database')
def do_log_tail(cs, args):
    """Display log entries for instance."""
    try:
        instance = _find_instance(cs, args.instance)
        log_gen = cs.instances.log_generator(instance, args.log_name, args.lines)
        for log_part in log_gen():
            print(log_part, end='')
    except exceptions.GuestLogNotFoundError:
        print(NO_LOG_FOUND_ERROR % {'log_name': args.log_name, 'instance': instance})
    except Exception as ex:
        error_msg = ex.message.split('\n')
        print(error_msg[0])