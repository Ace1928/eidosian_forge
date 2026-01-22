from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from magnumclient.v1 import basemodels
@utils.deprecation_map(DEPRECATING_PARAMS)
@utils.arg('positional_name', metavar='<name>', nargs='?', default=None, help=_('Name of the cluster template to create.'))
@utils.arg('--name', metavar='<name>', default=None, help=_('Name of the cluster template to create. %s') % utils.NAME_DEPRECATION_HELP)
@utils.arg('--image-id', dest='image', required=True, metavar='<image>', help=utils.deprecation_message('The name or UUID of the base image to customize for the Cluster.', 'image'))
@utils.arg('--image', dest='image', required=True, metavar='<image>', help=_('The name or UUID of the base image to customize for the Cluster.'))
@utils.arg('--keypair-id', dest='keypair', metavar='<keypair>', help=utils.deprecation_message('The name of the SSH keypair to load into the Cluster nodes.', 'keypair'))
@utils.arg('--keypair', dest='keypair', metavar='<keypair>', help=_('The name of the SSH keypair to load into the Cluster nodes.'))
@utils.arg('--external-network-id', dest='external_network', required=True, metavar='<external-network>', help=utils.deprecation_message('The external Neutron network name or UUID to connect to this Cluster Template.', 'external-network'))
@utils.arg('--external-network', dest='external_network', required=True, metavar='<external-network>', help=_('The external Neutron network name or UUID to connect to this Cluster Template.'))
@utils.arg('--coe', required=True, metavar='<coe>', help=_('Specify the Container Orchestration Engine to use.'))
@utils.arg('--fixed-network', metavar='<fixed-network>', help=_('The private Neutron network name to connect to this Cluster model.'))
@utils.arg('--fixed-subnet', metavar='<fixed-subnet>', help=_('The private Neutron subnet name to connect to Cluster.'))
@utils.arg('--network-driver', metavar='<network-driver>', help=_('The network driver name for instantiating container networks.'))
@utils.arg('--volume-driver', metavar='<volume-driver>', help=_('The volume driver name for instantiating container volume.'))
@utils.arg('--dns-nameserver', metavar='<dns-nameserver>', default='8.8.8.8', help=_('The DNS nameserver to use for this cluster template.'))
@utils.arg('--flavor-id', dest='flavor', metavar='<flavor>', default='m1.medium', help=utils.deprecation_message('The nova flavor name or UUID to use when launching the Cluster.', 'flavor'))
@utils.arg('--flavor', dest='flavor', metavar='<flavor>', default='m1.medium', help=_('The nova flavor name or UUID to use when launching the Cluster.'))
@utils.arg('--master-flavor-id', dest='master_flavor', metavar='<master-flavor>', help=utils.deprecation_message('The nova flavor name or UUID to use when launching the master node of the Cluster.', 'master-flavor'))
@utils.arg('--master-flavor', dest='master_flavor', metavar='<master-flavor>', help=_('The nova flavor name or UUID to use when launching the master node of the Cluster.'))
@utils.arg('--docker-volume-size', metavar='<docker-volume-size>', type=int, help=_('Specify the number of size in GB for the docker volume to use.'))
@utils.arg('--docker-storage-driver', metavar='<docker-storage-driver>', default='devicemapper', help=_('Select a docker storage driver. Supported: devicemapper, overlay. Default: devicemapper'))
@utils.arg('--http-proxy', metavar='<http-proxy>', help=_('The http_proxy address to use for nodes in Cluster.'))
@utils.arg('--https-proxy', metavar='<https-proxy>', help=_('The https_proxy address to use for nodes in Cluster.'))
@utils.arg('--no-proxy', metavar='<no-proxy>', help=_('The no_proxy address to use for nodes in Cluster.'))
@utils.arg('--labels', metavar='<KEY1=VALUE1,KEY2=VALUE2;KEY3=VALUE3...>', action='append', default=[], help=_('Arbitrary labels in the form of key=value pairs to associate with a cluster template. May be used multiple times.'))
@utils.arg('--tls-disabled', action='store_true', default=False, help=_('Disable TLS in the Cluster.'))
@utils.arg('--public', action='store_true', default=False, help=_('Make cluster template public.'))
@utils.arg('--registry-enabled', action='store_true', default=False, help=_('Enable docker registry in the Cluster'))
@utils.arg('--server-type', metavar='<server-type>', default='vm', help=_('Specify the server type to be used for example vm. For this release default server type will be vm.'))
@utils.arg('--master-lb-enabled', action='store_true', default=False, help=_('Indicates whether created Clusters should have a load balancer for master nodes or not.'))
@utils.arg('--floating-ip-enabled', action='append_const', const=True, default=[], dest='floating_ip_enabled', help=_('Indicates whether created Clusters should have a floating ip.'))
@utils.arg('--floating-ip-disabled', action='append_const', const=False, default=[], dest='floating_ip_enabled', help=_('Disables floating ip creation on the new Cluster'))
@utils.arg('--insecure-registry', metavar='<insecure-registry>', help='url of docker registry')
@utils.arg('--hidden', action='store_true', default=False, help=_('Make cluster template hidden.'))
@utils.arg('--visible', dest='hidden', action='store_false', help=_('Make cluster template visible.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_template_create(cs, args):
    """Create a cluster template."""
    args.command = 'cluster-template-create'
    utils.validate_name_args(args.positional_name, args.name)
    opts = {}
    opts['name'] = args.positional_name or args.name
    opts['flavor_id'] = args.flavor
    opts['master_flavor_id'] = args.master_flavor
    opts['image_id'] = args.image
    opts['keypair_id'] = args.keypair
    opts['external_network_id'] = args.external_network
    opts['fixed_network'] = args.fixed_network
    opts['fixed_subnet'] = args.fixed_subnet
    opts['network_driver'] = args.network_driver
    opts['volume_driver'] = args.volume_driver
    opts['dns_nameserver'] = args.dns_nameserver
    opts['docker_volume_size'] = args.docker_volume_size
    opts['docker_storage_driver'] = args.docker_storage_driver
    opts['coe'] = args.coe
    opts['http_proxy'] = args.http_proxy
    opts['https_proxy'] = args.https_proxy
    opts['no_proxy'] = args.no_proxy
    opts['labels'] = magnum_utils.handle_labels(args.labels)
    opts['tls_disabled'] = args.tls_disabled
    opts['public'] = args.public
    opts['registry_enabled'] = args.registry_enabled
    opts['server_type'] = args.server_type
    opts['master_lb_enabled'] = args.master_lb_enabled
    opts['insecure_registry'] = args.insecure_registry
    opts['hidden'] = args.hidden
    if len(args.floating_ip_enabled) > 1:
        raise InvalidAttribute('--floating-ip-enabled and --floating-ip-disabled are mutually exclusive and should be specified only once.')
    elif len(args.floating_ip_enabled) == 1:
        opts['floating_ip_enabled'] = args.floating_ip_enabled[0]
    cluster_template = cs.cluster_templates.create(**opts)
    _show_cluster_template(cluster_template)