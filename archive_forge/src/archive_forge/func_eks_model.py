import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def eks_model(name):
    eks_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(eks_data))
    return eks_models.get_waiter(name)