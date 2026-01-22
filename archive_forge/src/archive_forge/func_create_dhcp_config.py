from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def create_dhcp_config(module):
    """
    Convert provided parameters into a DhcpConfigurations list that conforms to what the API returns:
    https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeDhcpOptions.html
        [{'Key': 'domain-name',
         'Values': [{'Value': 'us-west-2.compute.internal'}]},
        {'Key': 'domain-name-servers',
         'Values': [{'Value': 'AmazonProvidedDNS'}]},
         ...],
    """
    new_config = []
    params = module.params
    if params['domain_name'] is not None:
        new_config.append({'Key': 'domain-name', 'Values': [{'Value': params['domain_name']}]})
    if params['dns_servers'] is not None:
        dns_server_list = []
        for server in params['dns_servers']:
            dns_server_list.append({'Value': server})
        new_config.append({'Key': 'domain-name-servers', 'Values': dns_server_list})
    if params['ntp_servers'] is not None:
        ntp_server_list = []
        for server in params['ntp_servers']:
            ntp_server_list.append({'Value': server})
        new_config.append({'Key': 'ntp-servers', 'Values': ntp_server_list})
    if params['netbios_name_servers'] is not None:
        netbios_server_list = []
        for server in params['netbios_name_servers']:
            netbios_server_list.append({'Value': server})
        new_config.append({'Key': 'netbios-name-servers', 'Values': netbios_server_list})
    if params['netbios_node_type'] is not None:
        new_config.append({'Key': 'netbios-node-type', 'Values': params['netbios_node_type']})
    return new_config