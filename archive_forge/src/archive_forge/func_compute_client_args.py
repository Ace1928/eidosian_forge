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
def compute_client_args(self, service_model, client_config, endpoint_bridge, region_name, endpoint_url, is_secure, scoped_config):
    service_name = service_model.endpoint_prefix
    protocol = service_model.metadata['protocol']
    parameter_validation = True
    if client_config and (not client_config.parameter_validation):
        parameter_validation = False
    elif scoped_config:
        raw_value = scoped_config.get('parameter_validation')
        if raw_value is not None:
            parameter_validation = ensure_boolean(raw_value)
    s3_config = self.compute_s3_config(client_config)
    configured_endpoint_url = self._compute_configured_endpoint_url(client_config=client_config, endpoint_url=endpoint_url)
    endpoint_config = self._compute_endpoint_config(service_name=service_name, region_name=region_name, endpoint_url=configured_endpoint_url, is_secure=is_secure, endpoint_bridge=endpoint_bridge, s3_config=s3_config)
    endpoint_variant_tags = endpoint_config['metadata'].get('tags', [])
    preliminary_ua_string = self._session_ua_creator.with_client_config(client_config).to_string()
    config_kwargs = dict(region_name=endpoint_config['region_name'], signature_version=endpoint_config['signature_version'], user_agent=preliminary_ua_string)
    if 'dualstack' in endpoint_variant_tags:
        config_kwargs.update(use_dualstack_endpoint=True)
    if 'fips' in endpoint_variant_tags:
        config_kwargs.update(use_fips_endpoint=True)
    if client_config is not None:
        config_kwargs.update(connect_timeout=client_config.connect_timeout, read_timeout=client_config.read_timeout, max_pool_connections=client_config.max_pool_connections, proxies=client_config.proxies, proxies_config=client_config.proxies_config, retries=client_config.retries, client_cert=client_config.client_cert, inject_host_prefix=client_config.inject_host_prefix, tcp_keepalive=client_config.tcp_keepalive, user_agent_extra=client_config.user_agent_extra, user_agent_appid=client_config.user_agent_appid, request_min_compression_size_bytes=client_config.request_min_compression_size_bytes, disable_request_compression=client_config.disable_request_compression, client_context_params=client_config.client_context_params)
    self._compute_retry_config(config_kwargs)
    self._compute_connect_timeout(config_kwargs)
    self._compute_user_agent_appid_config(config_kwargs)
    self._compute_request_compression_config(config_kwargs)
    s3_config = self.compute_s3_config(client_config)
    is_s3_service = self._is_s3_service(service_name)
    if is_s3_service and 'dualstack' in endpoint_variant_tags:
        if s3_config is None:
            s3_config = {}
        s3_config['use_dualstack_endpoint'] = True
    return {'service_name': service_name, 'parameter_validation': parameter_validation, 'configured_endpoint_url': configured_endpoint_url, 'endpoint_config': endpoint_config, 'protocol': protocol, 'config_kwargs': config_kwargs, 's3_config': s3_config, 'socket_options': self._compute_socket_options(scoped_config, client_config)}