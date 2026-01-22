import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def ec2_model(name):
    ec2_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(ec2_data))
    return ec2_models.get_waiter(name)