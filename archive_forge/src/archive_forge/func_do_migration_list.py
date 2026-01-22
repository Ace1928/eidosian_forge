import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@api_versions.wraps('2.66')
@utils.arg('--instance-uuid', dest='instance_uuid', metavar='<instance_uuid>', help=_('Fetch migrations for the given instance.'))
@utils.arg('--host', dest='host', metavar='<host>', help=_('Fetch migrations for the given source or destination host.'))
@utils.arg('--status', dest='status', metavar='<status>', help=_('Fetch migrations for the given status.'))
@utils.arg('--migration-type', dest='migration_type', metavar='<migration_type>', help=_('Filter migrations by type. Valid values are: evacuation, live-migration, migration (cold), resize'))
@utils.arg('--source-compute', dest='source_compute', metavar='<source_compute>', help=_('Filter migrations by source compute host name.'))
@utils.arg('--marker', dest='marker', metavar='<marker>', default=None, help=_('The last migration of the previous page; displays list of migrations after "marker". Note that the marker is the migration UUID.'))
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_('Maximum number of migrations to display. Note that there is a configurable max limit on the server, and the limit that is used will be the minimum of what is requested here and what is configured in the server.'))
@utils.arg('--changes-since', dest='changes_since', metavar='<changes_since>', default=None, help=_('List only migrations changed later or equal to a certain point of time. The provided time should be an ISO 8061 formatted time. e.g. 2016-03-04T06:27:59Z .'))
@utils.arg('--changes-before', dest='changes_before', metavar='<changes_before>', default=None, help=_('List only migrations changed earlier or equal to a certain point of time. The provided time should be an ISO 8061 formatted time. e.g. 2016-03-04T06:27:59Z .'), start_version='2.66')
@utils.arg('--project-id', dest='project_id', metavar='<project_id>', default=None, help=_('Filter the migrations by the given project ID.'), start_version='2.80')
@utils.arg('--user-id', dest='user_id', metavar='<user_id>', default=None, help=_('Filter the migrations by the given user ID.'), start_version='2.80')
def do_migration_list(cs, args):
    """Print a list of migrations."""
    if args.changes_since:
        try:
            timeutils.parse_isotime(args.changes_since)
        except ValueError:
            raise exceptions.CommandError(_('Invalid changes-since value: %s') % args.changes_since)
    if args.changes_before:
        try:
            timeutils.parse_isotime(args.changes_before)
        except ValueError:
            raise exceptions.CommandError(_('Invalid changes-before value: %s') % args.changes_before)
    kwargs = dict(instance_uuid=args.instance_uuid, marker=args.marker, limit=args.limit, changes_since=args.changes_since, changes_before=args.changes_before, migration_type=args.migration_type, source_compute=args.source_compute)
    if cs.api_version >= api_versions.APIVersion('2.80'):
        kwargs['project_id'] = args.project_id
        kwargs['user_id'] = args.user_id
    migrations = cs.migrations.list(args.host, args.status, **kwargs)
    _print_migrations(cs, migrations)