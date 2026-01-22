from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def build_route_spec(self, service_name, namespace, port=None, wildcard_policy=None, hostname=None, path=None, termination_type=None, tls_insecure_policy=None, tls_ca_cert=None, tls_cert=None, tls_key=None, tls_dest_ca_cert=None):
    v1_services = self.find_resource('Service', 'v1', fail=True)
    try:
        target_service = v1_services.get(name=service_name, namespace=namespace)
    except NotFoundError:
        if not port:
            self.fail_json(msg="You need to provide the 'port' argument when exposing a non-existent service")
        target_service = None
    except DynamicApiError as exc:
        self.fail_json(msg='Failed to retrieve service to be exposed: {0}'.format(exc.body), error=exc.status, status=exc.status, reason=exc.reason)
    except Exception as exc:
        self.fail_json(msg='Failed to retrieve service to be exposed: {0}'.format(to_native(exc)), error='', status='', reason='')
    route_spec = {'tls': {}, 'to': {'kind': 'Service', 'name': service_name}, 'port': {'targetPort': self.set_port(target_service, port)}, 'wildcardPolicy': wildcard_policy}
    if termination_type:
        route_spec['tls'] = dict(termination=termination_type.capitalize())
        if tls_insecure_policy:
            if termination_type == 'edge':
                route_spec['tls']['insecureEdgeTerminationPolicy'] = tls_insecure_policy.capitalize()
            elif termination_type == 'passthrough':
                if tls_insecure_policy != 'redirect':
                    self.fail_json("'redirect' is the only supported insecureEdgeTerminationPolicy for passthrough routes")
                route_spec['tls']['insecureEdgeTerminationPolicy'] = tls_insecure_policy.capitalize()
            elif termination_type == 'reencrypt':
                self.fail_json("'tls.insecure_policy' is not supported with reencrypt routes")
        else:
            route_spec['tls']['insecureEdgeTerminationPolicy'] = None
        if tls_ca_cert:
            if termination_type == 'passthrough':
                self.fail_json("'tls.ca_certificate' is not supported with passthrough routes")
            route_spec['tls']['caCertificate'] = tls_ca_cert
        if tls_cert:
            if termination_type == 'passthrough':
                self.fail_json("'tls.certificate' is not supported with passthrough routes")
            route_spec['tls']['certificate'] = tls_cert
        if tls_key:
            if termination_type == 'passthrough':
                self.fail_json("'tls.key' is not supported with passthrough routes")
            route_spec['tls']['key'] = tls_key
        if tls_dest_ca_cert:
            if termination_type != 'reencrypt':
                self.fail_json("'destination_certificate' is only valid for reencrypt routes")
            route_spec['tls']['destinationCACertificate'] = tls_dest_ca_cert
    else:
        route_spec['tls'] = None
    if hostname:
        route_spec['host'] = hostname
    if path:
        route_spec['path'] = path
    return route_spec