from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def create_credential_config(args, config_type):
    """Creates the byoid credential config based on CLI arguments."""
    enable_mtls = getattr(args, 'enable_mtls', False)
    universe_domain_property = properties.VALUES.core.universe_domain
    if getattr(args, 'universe_domain', None):
        universe_domain = args.universe_domain
    elif universe_domain_property.IsExplicitlySet():
        universe_domain = universe_domain_property.Get()
    else:
        universe_domain = properties.VALUES.core.universe_domain.default
    token_endpoint_builder = StsEndpoints(enable_mtls=enable_mtls, universe_domain=universe_domain)
    try:
        generator = get_generator(args, config_type)
        output = {'universe_domain': universe_domain, 'type': 'external_account', 'audience': '//iam.googleapis.com/' + args.audience, 'subject_token_type': generator.get_token_type(args.subject_token_type), 'token_url': token_endpoint_builder.token_url, 'credential_source': generator.get_source(args)}
        if config_type is ConfigType.WORKFORCE_POOLS:
            output['workforce_pool_user_project'] = args.workforce_pool_user_project
        if args.service_account:
            sa_endpoint_builder = IamEndpoints(args.service_account, enable_mtls=enable_mtls, universe_domain=universe_domain)
            output['service_account_impersonation_url'] = sa_endpoint_builder.impersonation_url
            service_account_impersonation = {}
            if args.service_account_token_lifetime_seconds:
                service_account_impersonation['token_lifetime_seconds'] = args.service_account_token_lifetime_seconds
                output['service_account_impersonation'] = service_account_impersonation
        else:
            output['token_info_url'] = token_endpoint_builder.token_info_url
        files.WriteFileContents(args.output_file, json.dumps(output, indent=2))
        log.CreatedResource(args.output_file, RESOURCE_TYPE)
    except GeneratorError as cce:
        log.CreatedResource(args.output_file, RESOURCE_TYPE, failed=cce.message)