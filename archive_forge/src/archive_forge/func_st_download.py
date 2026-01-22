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
def st_download(parser, args, output_manager, return_parser=False):
    parser.add_argument('-a', '--all', action='store_true', dest='yes_all', default=False, help='Indicates that you really want to download everything in the account.')
    parser.add_argument('-m', '--marker', dest='marker', default='', help='Marker to use when starting a container or account download.')
    parser.add_argument('-p', '--prefix', dest='prefix', help='Only download items beginning with the <prefix>.')
    parser.add_argument('-o', '--output', dest='out_file', help='For a single download, stream the output to <out_file>. Specifying "-" as <out_file> will redirect to stdout.')
    parser.add_argument('-D', '--output-dir', dest='out_directory', help='An optional directory to which to store objects. By default, all objects are recreated in the current directory.')
    parser.add_argument('-r', '--remove-prefix', action='store_true', dest='remove_prefix', default=False, help='An optional flag for --prefix <prefix>, use this option to download items without <prefix>.')
    parser.add_argument('--object-threads', type=int, default=10, help='Number of threads to use for downloading objects. Its value must be a positive integer. Default is 10.')
    parser.add_argument('--container-threads', type=int, default=10, help='Number of threads to use for downloading containers. Its value must be a positive integer. Default is 10.')
    parser.add_argument('--no-download', action='store_true', default=False, help="Perform download(s), but don't actually write anything to disk.")
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a customized request header to the query, like "Range" or "If-Match". This option may be repeated. Example: --header "content-type:text/plain"')
    parser.add_argument('--skip-identical', action='store_true', dest='skip_identical', default=False, help='Skip downloading files that are identical on both sides.')
    parser.add_argument('--version-id', action='store', default=None, help='Download a specific version of a versioned object')
    parser.add_argument('--ignore-checksum', action='store_false', dest='checksum', default=True, help='Turn off checksum validation for downloads.')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle', default=True, help='By default, download order is randomised in order to reduce the load on individual drives when multiple clients are executed simultaneously to download the same set of objects (e.g. a nightly automated download script to multiple servers). Enable this option to submit download jobs to the thread pool in the order they are listed in the object store.')
    parser.add_argument('--ignore-mtime', action='store_true', dest='ignore_mtime', default=False, help='By default, the object-meta-mtime header is used to store the access and modified timestamp for the downloaded file. With this option, the header is ignored and the timestamps are created freshly.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if options['out_file'] == '-':
        options['verbose'] = 0
    if options['out_file'] and len(args) != 2:
        exit('-o option only allowed for single file downloads')
    if not options['prefix']:
        options['remove_prefix'] = False
    if options['out_directory'] and len(args) == 2:
        exit('Please use -o option for single file downloads and renames')
    if not args and (not options['yes_all']) or (args and options['yes_all']):
        output_manager.error('Usage: %s download %s\n%s', BASENAME, st_download_options, st_download_help)
        return
    if options['version_id'] and len(args) < 2:
        exit('--version-id option only allowed for object downloads')
    if options['object_threads'] <= 0:
        output_manager.error('ERROR: option --object-threads should be a positive integer.\n\nUsage: %s download %s\n%s', BASENAME, st_download_options, st_download_help)
        return
    if options['container_threads'] <= 0:
        output_manager.error('ERROR: option --container-threads should be a positive integer.\n\nUsage: %s download %s\n%s', BASENAME, st_download_options, st_download_help)
        return
    options['object_dd_threads'] = options['object_threads']
    with SwiftService(options=options) as swift:
        try:
            if not args:
                down_iter = swift.download()
            else:
                container = args[0]
                if '/' in container:
                    output_manager.error("WARNING: / in container name; you might have meant '%s' instead of '%s'." % (container.replace('/', ' ', 1), container))
                    return
                objects = args[1:]
                if not objects:
                    down_iter = swift.download(container)
                else:
                    down_iter = swift.download(container, objects)
            for down in down_iter:
                if options['out_file'] == '-' and 'contents' in down:
                    contents = down['contents']
                    for chunk in contents:
                        output_manager.print_raw(chunk)
                elif down['success']:
                    if options['verbose']:
                        start_time = down['start_time']
                        headers_receipt = down['headers_receipt'] - start_time
                        auth_time = down['auth_end_time'] - start_time
                        finish_time = down['finish_time']
                        read_length = down['read_length']
                        attempts = down['attempts']
                        total_time = finish_time - start_time
                        down_time = total_time - auth_time
                        _mega = 1000000
                        if down['pseudodir']:
                            time_str = 'auth %.3fs, headers %.3fs, total %.3fs, pseudo' % (auth_time, headers_receipt, total_time)
                        else:
                            speed = float(read_length) / down_time / _mega
                            time_str = 'auth %.3fs, headers %.3fs, total %.3fs, %.3f MB/s' % (auth_time, headers_receipt, total_time, speed)
                        path = down['path']
                        if attempts > 1:
                            output_manager.print_msg('%s [%s after %d attempts]', path, time_str, attempts)
                        else:
                            output_manager.print_msg('%s [%s]', path, time_str)
                else:
                    error = down['error']
                    path = down['path']
                    container = down['container']
                    obj = down['object']
                    if isinstance(error, ClientException):
                        if error.http_status == 304 and options['skip_identical']:
                            output_manager.print_msg("Skipped identical file '%s'", path)
                            continue
                        if error.http_status == 404:
                            output_manager.error("Object '%s/%s' not found", container, obj)
                            continue
                    output_manager.error("Error downloading object '%s/%s': %s", container, obj, error)
        except SwiftError as e:
            output_manager.error(e.value)
        except Exception as e:
            output_manager.error(e)