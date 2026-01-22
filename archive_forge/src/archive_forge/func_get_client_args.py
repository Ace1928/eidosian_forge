import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def get_client_args(self, service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token=None, endpoints_ruleset_data=None, partition_data=None):
    final_args = self.compute_client_args(service_model, client_config, endpoint_bridge, region_name, endpoint_url, is_secure, scoped_config)
    service_name = final_args['service_name']
    parameter_validation = final_args['parameter_validation']
    endpoint_config = final_args['endpoint_config']
    protocol = final_args['protocol']
    config_kwargs = final_args['config_kwargs']
    s3_config = final_args['s3_config']
    partition = endpoint_config['metadata'].get('partition', None)
    socket_options = final_args['socket_options']
    configured_endpoint_url = final_args['configured_endpoint_url']
    signing_region = endpoint_config['signing_region']
    endpoint_region_name = endpoint_config['region_name']
    event_emitter = copy.copy(self._event_emitter)
    signer = RequestSigner(service_model.service_id, signing_region, endpoint_config['signing_name'], endpoint_config['signature_version'], credentials, event_emitter, auth_token)
    config_kwargs['s3'] = s3_config
    new_config = Config(**config_kwargs)
    endpoint_creator = EndpointCreator(event_emitter)
    endpoint = endpoint_creator.create_endpoint(service_model, region_name=endpoint_region_name, endpoint_url=endpoint_config['endpoint_url'], verify=verify, response_parser_factory=self._response_parser_factory, max_pool_connections=new_config.max_pool_connections, proxies=new_config.proxies, timeout=(new_config.connect_timeout, new_config.read_timeout), socket_options=socket_options, client_cert=new_config.client_cert, proxies_config=new_config.proxies_config)
    serializer = botocore.serialize.create_serializer(protocol, parameter_validation)
    response_parser = botocore.parsers.create_parser(protocol)
    ruleset_resolver = self._build_endpoint_resolver(endpoints_ruleset_data, partition_data, client_config, service_model, endpoint_region_name, region_name, configured_endpoint_url, endpoint, is_secure, endpoint_bridge, event_emitter)
    client_ua_creator = self._session_ua_creator.with_client_config(new_config)
    supplied_ua = client_config.user_agent if client_config else None
    new_config._supplied_user_agent = supplied_ua
    return {'serializer': serializer, 'endpoint': endpoint, 'response_parser': response_parser, 'event_emitter': event_emitter, 'request_signer': signer, 'service_model': service_model, 'loader': self._loader, 'client_config': new_config, 'partition': partition, 'exceptions_factory': self._exceptions_factory, 'endpoint_ruleset_resolver': ruleset_resolver, 'user_agent_creator': client_ua_creator}