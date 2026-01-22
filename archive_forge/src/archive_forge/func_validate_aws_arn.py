import re
def validate_aws_arn(arn, partition=None, service=None, region=None, account_id=None, resource=None, resource_type=None, resource_id=None):
    details = parse_aws_arn(arn)
    if not details:
        return False
    if partition and details.get('partition') != partition:
        return False
    if service and details.get('service') != service:
        return False
    if region and details.get('region') != region:
        return False
    if account_id and details.get('account_id') != account_id:
        return False
    if resource and details.get('resource') != resource:
        return False
    if resource_type and details.get('resource_type') != resource_type:
        return False
    if resource_id and details.get('resource_id') != resource_id:
        return False
    return True