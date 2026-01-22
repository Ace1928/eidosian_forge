import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--sort-by', metavar='<SORT BY FIELDS>', help='Fields to sort by as a comma separated list. Valid values are id, name, type, address, created_at, updated_at. Fields may be followed by "asc" or "desc", ex "address desc", to set the direction of sorting.')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
def do_notification_list(mc, args):
    """List notifications for this tenant."""
    fields = {}
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.sort_by:
        sort_by = args.sort_by.split(',')
        for field in sort_by:
            field_values = field.lower().split()
            if len(field_values) > 2:
                print('Invalid sort_by value {}'.format(field))
            if field_values[0] not in allowed_notification_sort_by:
                print('Sort-by field name {} is not in [{}]'.format(field_values[0], allowed_notification_sort_by))
                return
            if len(field_values) > 1 and field_values[1] not in ['asc', 'desc']:
                print('Invalid value {}, must be asc or desc'.format(field_values[1]))
        fields['sort_by'] = args.sort_by
    try:
        notification = mc.notifications.list(**fields)
    except osc_exc.ClientException as he:
        raise osc_exc.CommandError('ClientException code=%s message=%s' % (he.code, he.message))
    else:
        if args.json:
            print(utils.json_formatter(notification))
            return
        cols = ['name', 'id', 'type', 'address', 'period']
        formatters = {'name': lambda x: x['name'], 'id': lambda x: x['id'], 'type': lambda x: x['type'], 'address': lambda x: x['address'], 'period': lambda x: x['period']}
        if isinstance(notification, list):
            utils.print_list(notification, cols, formatters=formatters)
        else:
            notif_list = list()
            notif_list.append(notification)
            utils.print_list(notif_list, cols, formatters=formatters)