import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def elbv2_model(name):
    elbv2_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(elbv2_data))
    return elbv2_models.get_waiter(name)