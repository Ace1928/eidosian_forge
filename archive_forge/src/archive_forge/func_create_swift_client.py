import copy
import uuid
from swiftclient import client as sc
from swiftclient import utils as swiftclient_utils
from urllib import parse as urlparse
from heatclient._i18n import _
from heatclient import exc
from heatclient.v1 import software_configs
def create_swift_client(auth, session, args):
    auth_token = auth.get_token(session)
    endpoint = auth.get_endpoint(session, service_type='object-store', region_name=args.os_region_name)
    project_name = args.os_project_name or args.os_tenant_name
    swift_args = {'auth_version': '2.0', 'tenant_name': project_name, 'user': args.os_username, 'key': None, 'authurl': None, 'preauthtoken': auth_token, 'preauthurl': endpoint, 'cacert': args.os_cacert, 'insecure': args.insecure}
    return sc.Connection(**swift_args)