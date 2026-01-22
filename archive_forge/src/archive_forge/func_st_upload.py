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
def st_upload(parser, args, output_manager, return_parser=False):
    DEFAULT_STDIN_SEGMENT = 10 * 1024 * 1024
    parser.add_argument('-c', '--changed', action='store_true', dest='changed', default=False, help='Only upload files that have changed since the last upload.')
    parser.add_argument('--skip-identical', action='store_true', dest='skip_identical', default=False, help='Skip uploading files that are identical on both sides.')
    parser.add_argument('--skip-container-put', action='store_true', dest='skip_container_put', default=False, help="Assume all necessary containers already exist; don't automatically try to create them.")
    parser.add_argument('-S', '--segment-size', dest='segment_size', help='Upload files in segments no larger than <size> (in Bytes) and then create a "manifest" file that will download all the segments as if it were the original file. Sizes may also be expressed as bytes with the B suffix, kilobytes with the K suffix, megabytes with the M suffix or gigabytes with the G suffix.')
    parser.add_argument('-C', '--segment-container', dest='segment_container', help='Upload the segments into the specified container. If not specified, the segments will be uploaded to a <container>_segments container to not pollute the main <container> listings.')
    parser.add_argument('--leave-segments', action='store_true', dest='leave_segments', default=False, help='Indicates that you want the older segments of manifest objects left alone (in the case of overwrites).')
    parser.add_argument('--object-threads', type=int, default=10, help='Number of threads to use for uploading full objects. Its value must be a positive integer. Default is 10.')
    parser.add_argument('--segment-threads', type=int, default=10, help='Number of threads to use for uploading object segments. Its value must be a positive integer. Default is 10.')
    parser.add_argument('-m', '--meta', action='append', dest='meta', default=[], help='Sets a meta data item. This option may be repeated. Example: -m Color:Blue -m Size:Large')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Set request headers with the syntax header:value.  This option may be repeated. Example: -H "content-type:text/plain" -H "Content-Length: 4000"')
    parser.add_argument('--use-slo', action='store_true', default=None, help='When used in conjunction with --segment-size, it will create a Static Large Object.')
    parser.add_argument('--use-dlo', action='store_false', dest='use_slo', default=None, help='When used in conjunction with --segment-size, it will create a Dynamic Large Object.')
    parser.add_argument('--object-name', dest='object_name', help='Upload file and name object to <object-name> or upload dir and use <object-name> as object prefix instead of folder name.')
    parser.add_argument('--ignore-checksum', dest='checksum', default=True, action='store_false', help='Turn off checksum validation for uploads.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if len(args) < 2:
        output_manager.error('Usage: %s upload %s\n%s', BASENAME, st_upload_options, st_upload_help)
        return
    else:
        container = args[0]
        files = args[1:]
        from_stdin = '-' in files
        if from_stdin and len(files) > 1:
            output_manager.error('upload from stdin cannot be used along with other files')
            return
    if options['object_name'] is not None:
        if len(files) > 1:
            output_manager.error('object-name only be used with 1 file or dir')
            return
        else:
            orig_path = files[0]
    elif from_stdin:
        output_manager.error('object-name must be specified with uploads from stdin')
        return
    if options['segment_size']:
        try:
            int(options['segment_size'])
        except ValueError:
            try:
                size_mod = 'BKMG'.index(options['segment_size'][-1].upper())
                multiplier = int(options['segment_size'][:-1])
            except ValueError:
                output_manager.error('Invalid segment size')
                return
            options['segment_size'] = str(1024 ** size_mod * multiplier)
        if int(options['segment_size']) <= 0:
            output_manager.error('segment-size should be positive')
            return
    if options['object_threads'] <= 0:
        output_manager.error('ERROR: option --object-threads should be a positive integer.\n\nUsage: %s upload %s\n%s', BASENAME, st_upload_options, st_upload_help)
        return
    if options['segment_threads'] <= 0:
        output_manager.error('ERROR: option --segment-threads should be a positive integer.\n\nUsage: %s upload %s\n%s', BASENAME, st_upload_options, st_upload_help)
        return
    if from_stdin:
        if options['use_slo'] is None:
            options['use_slo'] = True
        if not options['segment_size']:
            options['segment_size'] = DEFAULT_STDIN_SEGMENT
    options['object_uu_threads'] = options['object_threads']
    with SwiftService(options=options) as swift:
        try:
            objs = []
            dir_markers = []
            for f in files:
                if f == '-':
                    fd = io.open(stdin.fileno(), mode='rb')
                    objs.append(SwiftUploadObject(fd, object_name=options['object_name']))
                    break
                if isfile(f):
                    objs.append(f)
                elif isdir(f):
                    for _dir, _ds, _fs in walk(f):
                        if not _ds + _fs:
                            dir_markers.append(_dir)
                        else:
                            objs.extend([join(_dir, _f) for _f in _fs])
                else:
                    output_manager.error("Local file '%s' not found" % f)
            if options['object_name'] is not None and (not from_stdin):
                objs = [SwiftUploadObject(o, object_name=o.replace(orig_path, options['object_name'], 1)) for o in objs]
                dir_markers = [SwiftUploadObject(None, object_name=d.replace(orig_path, options['object_name'], 1), options={'dir_marker': True}) for d in dir_markers]
            for r in swift.upload(container, objs + dir_markers):
                if r['success']:
                    if options['verbose']:
                        if 'attempts' in r and r['attempts'] > 1:
                            if 'object' in r:
                                output_manager.print_msg('%s [after %d attempts]' % (r['object'], r['attempts']))
                        elif 'object' in r:
                            output_manager.print_msg(r['object'])
                        elif 'for_object' in r:
                            output_manager.print_msg('%s segment %s' % (r['for_object'], r['segment_index']))
                else:
                    error = r['error']
                    if 'action' in r and r['action'] == 'create_container':
                        if isinstance(error, ClientException):
                            if r['headers'] and 'X-Storage-Policy' in r['headers']:
                                msg = ' with Storage Policy %s' % r['headers']['X-Storage-Policy'].strip()
                            else:
                                msg = ' '.join((str(x) for x in (error.http_status, error.http_reason)))
                                if error.http_response_content:
                                    if msg:
                                        msg += ': '
                                    msg += error.http_response_content.decode('utf8')[:60]
                                msg = ': %s' % msg
                        else:
                            msg = ': %s' % error
                        output_manager.warning("Warning: failed to create container '%s'%s", r['container'], msg)
                    else:
                        output_manager.error('%s' % error)
                        too_large = isinstance(error, ClientException) and error.http_status == 413
                        if too_large and options['verbose'] > 0:
                            output_manager.error('Consider using the --segment-size option to chunk the object')
        except SwiftError as e:
            output_manager.error(e.value)