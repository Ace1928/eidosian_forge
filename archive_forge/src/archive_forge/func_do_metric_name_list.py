import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--dimensions', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair used to specify a metric dimension. This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
@utils.arg('--tenant-id', metavar='<TENANT_ID>', help="Retrieve data for the specified tenant/project id instead of the tenant/project from the user's Keystone credentials.")
def do_metric_name_list(mc, args):
    """List names of metrics."""
    fields = {}
    if args.dimensions:
        fields['dimensions'] = utils.format_dimensions_query(args.dimensions)
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.tenant_id:
        fields['tenant_id'] = args.tenant_id
    try:
        metric_names = mc.metrics.list_names(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(metric_names))
            return
        if isinstance(metric_names, list):
            utils.print_list(metric_names, ['Name'], formatters={'Name': lambda x: x['name']})