import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def common_params_for_list(args, fields, field_labels):
    """Generate 'params' dict that is common for every 'list' command.

    :param args: arguments from command line.
    :param fields: possible fields for sorting.
    :param field_labels: possible field labels for sorting.
    :returns: a dict with params to pass to the client method.
    """
    params = {}
    if args.marker is not None:
        params['marker'] = args.marker
    if args.limit is not None:
        if args.limit < 0:
            raise exc.CommandError(_('Expected non-negative --limit, got %s') % args.limit)
        params['limit'] = args.limit
    if args.sort_key is not None:
        fields_map = dict(zip(field_labels, fields))
        fields_map.update(zip(fields, fields))
        try:
            sort_key = fields_map[args.sort_key]
        except KeyError:
            raise exc.CommandError(_('%(sort_key)s is an invalid field for sorting, valid values for --sort-key are: %(valid)s') % {'sort_key': args.sort_key, 'valid': list(fields_map)})
        params['sort_key'] = sort_key
    if args.sort_dir is not None:
        if args.sort_dir not in ('asc', 'desc'):
            raise exc.CommandError(_("%s is an invalid value for sort direction, valid values for --sort-dir are: 'asc', 'desc'") % args.sort_dir)
        params['sort_dir'] = args.sort_dir
    params['detail'] = args.detail
    requested_fields = args.fields[0] if args.fields else None
    if requested_fields is not None:
        params['fields'] = requested_fields
    return params