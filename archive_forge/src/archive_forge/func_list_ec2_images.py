from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_ec2_images(ec2_client, module, request_args):
    images = get_images(ec2_client, request_args)['Images']
    images = [camel_dict_to_snake_dict(image) for image in images]
    for image in images:
        try:
            image_id = image['image_id']
            image['tags'] = boto3_tag_list_to_ansible_dict(image.get('tags', []))
            if module.params.get('describe_image_attributes'):
                launch_permissions = get_image_attribute(ec2_client, image_id).get('LaunchPermissions', [])
                image['launch_permissions'] = [camel_dict_to_snake_dict(perm) for perm in launch_permissions]
        except is_boto3_error_code('AuthFailure'):
            pass
        except (ClientError, BotoCoreError) as err:
            raise AmiInfoFailure(err, 'Failed to describe AMI')
    images.sort(key=lambda e: e.get('creation_date', ''))
    return images