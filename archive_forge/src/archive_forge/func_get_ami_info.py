import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_ami_info(camel_image):
    image = camel_dict_to_snake_dict(camel_image)
    return dict(image_id=image.get('image_id'), state=image.get('state'), architecture=image.get('architecture'), block_device_mapping=get_block_device_mapping(image), creationDate=image.get('creation_date'), description=image.get('description'), hypervisor=image.get('hypervisor'), is_public=image.get('public'), location=image.get('image_location'), ownerId=image.get('owner_id'), root_device_name=image.get('root_device_name'), root_device_type=image.get('root_device_type'), virtualization_type=image.get('virtualization_type'), name=image.get('name'), tags=boto3_tag_list_to_ansible_dict(image.get('tags')), platform=image.get('platform'), enhanced_networking=image.get('ena_support'), image_owner_alias=image.get('image_owner_alias'), image_type=image.get('image_type'), kernel_id=image.get('kernel_id'), product_codes=image.get('product_codes'), ramdisk_id=image.get('ramdisk_id'), sriov_net_support=image.get('sriov_net_support'), state_reason=image.get('state_reason'), launch_permissions=image.get('launch_permissions'))