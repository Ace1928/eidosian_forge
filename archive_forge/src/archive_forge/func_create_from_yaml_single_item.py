import re
from os import path
from ruamel import yaml
from kubernetes import client
def create_from_yaml_single_item(k8s_client, yml_object, verbose=False, **kwargs):
    group, _, version = yml_object['apiVersion'].partition('/')
    if version == '':
        version = group
        group = 'core'
    group = ''.join(group.rsplit('.k8s.io', 1))
    group = ''.join((word.capitalize() for word in group.split('.')))
    fcn_to_call = '{0}{1}Api'.format(group, version.capitalize())
    k8s_api = getattr(client, fcn_to_call)(k8s_client)
    kind = yml_object['kind']
    kind = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', kind)
    kind = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', kind).lower()
    if hasattr(k8s_api, 'create_namespaced_{0}'.format(kind)):
        if 'namespace' in yml_object['metadata']:
            namespace = yml_object['metadata']['namespace']
            kwargs['namespace'] = namespace
        resp = getattr(k8s_api, 'create_namespaced_{0}'.format(kind))(body=yml_object, **kwargs)
    else:
        kwargs.pop('namespace', None)
        resp = getattr(k8s_api, 'create_{0}'.format(kind))(body=yml_object, **kwargs)
    if verbose:
        msg = '{0} created.'.format(kind)
        if hasattr(resp, 'status'):
            msg += " status='{0}'".format(str(resp.status))
        print(msg)