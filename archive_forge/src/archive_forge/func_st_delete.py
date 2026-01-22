import argparse
import getpass
import io
import json
import logging
import signal
import socket
import warnings
from os import environ, walk, _exit as os_exit
from os.path import isfile, isdir, join
from urllib.parse import unquote, urlparse
from sys import argv as sys_argv, exit, stderr, stdin
from time import gmtime, strftime
from swiftclient import RequestException
from swiftclient.utils import config_true_value, generate_temp_url, \
from swiftclient.multithreading import OutputManager
from swiftclient.exceptions import ClientException
from swiftclient import __version__ as client_version
from swiftclient.client import logger_settings as client_logger_settings, \
from swiftclient.service import SwiftService, SwiftError, \
from swiftclient.command_helpers import print_account_stats, \
def st_delete(parser, args, output_manager, return_parser=False):
    parser.add_argument('-a', '--all', action='store_true', dest='yes_all', default=False, help='Delete all containers and objects.')
    parser.add_argument('--versions', action='store_true', help='delete all versions')
    parser.add_argument('-p', '--prefix', dest='prefix', help='Only delete items beginning with <prefix>.')
    parser.add_argument('--version-id', action='store', default=None, help='Delete a specific version of a versioned object')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a custom request header to use for deleting objects or an entire container.')
    parser.add_argument('--leave-segments', action='store_true', dest='leave_segments', default=False, help='Do not delete segments of manifest objects.')
    parser.add_argument('--object-threads', type=int, default=10, help='Number of threads to use for deleting objects. Its value must be a positive integer. Default is 10.')
    parser.add_argument('--container-threads', type=int, default=10, help='Number of threads to use for deleting containers. Its value must be a positive integer. Default is 10.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if options['yes_all']:
        options['versions'] = True
    if not args and (not options['yes_all']) or (args and options['yes_all']):
        output_manager.error('Usage: %s delete %s\n%s', BASENAME, st_delete_options, st_delete_help)
        return
    if options['versions'] and len(args) >= 2:
        exit('--versions option not allowed for object deletes')
    if options['version_id'] and len(args) < 2:
        exit('--version-id option only allowed for object deletes')
    if options['object_threads'] <= 0:
        output_manager.error('ERROR: option --object-threads should be a positive integer.\n\nUsage: %s delete %s\n%s', BASENAME, st_delete_options, st_delete_help)
        return
    if options['container_threads'] <= 0:
        output_manager.error('ERROR: option --container-threads should be a positive integer.\n\nUsage: %s delete %s\n%s', BASENAME, st_delete_options, st_delete_help)
        return
    options['object_dd_threads'] = options['object_threads']
    with SwiftService(options=options) as swift:
        try:
            if not args:
                del_iter = swift.delete()
            else:
                container = args[0]
                if '/' in container:
                    output_manager.error("WARNING: / in container name; you might have meant '%s' instead of '%s'." % (container.replace('/', ' ', 1), container))
                    return
                objects = args[1:]
                if objects:
                    del_iter = swift.delete(container=container, objects=objects)
                else:
                    del_iter = swift.delete(container=container)
            for r in del_iter:
                c = r.get('container', '')
                o = r.get('object', '')
                a = ' [after {0} attempts]'.format(r.get('attempts')) if r.get('attempts', 1) > 1 else ''
                if r['action'] == 'bulk_delete':
                    if r['success']:
                        objs = r.get('objects', [])
                        for o, err in r.get('result', {}).get('Errors', []):
                            o = unquote(o)
                            output_manager.error('Error Deleting: {0}: {1}'.format(o[1:], err))
                            try:
                                objs.remove(o[len(c) + 2:])
                            except ValueError:
                                pass
                        for o in objs:
                            if options['yes_all']:
                                p = '{0}/{1}'.format(c, o)
                            else:
                                p = o
                            output_manager.print_msg('{0}{1}'.format(p, a))
                    else:
                        for o in r.get('objects', []):
                            output_manager.error('Error Deleting: {0}/{1}: {2}'.format(c, o, r['error']))
                elif r['success']:
                    if options['verbose']:
                        if r['action'] == 'delete_object':
                            if options['yes_all']:
                                p = '{0}/{1}'.format(c, o)
                            else:
                                p = o
                        elif r['action'] == 'delete_segment':
                            p = '{0}/{1}'.format(c, o)
                        elif r['action'] == 'delete_container':
                            p = c
                        output_manager.print_msg('{0}{1}'.format(p, a))
                else:
                    p = '{0}/{1}'.format(c, o) if o else c
                    output_manager.error('Error Deleting: {0}: {1}'.format(p, r['error']))
        except SwiftError as err:
            output_manager.error(err.value)