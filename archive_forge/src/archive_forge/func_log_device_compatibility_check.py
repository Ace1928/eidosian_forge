import itertools
from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
def log_device_compatibility_check(policy_name):
    """Logs a compatibility check if the devices support the policy.

  Currently only logs for the policy mixed_float16. A log is shown only the
  first time this function is called.

  Args:
    policy_name: The name of the dtype policy.
  """
    global _logged_compatibility_check
    if _logged_compatibility_check:
        return
    _logged_compatibility_check = True
    gpus = config.list_physical_devices('GPU')
    gpu_details_list = [config.get_device_details(g) for g in gpus]
    _log_device_compatibility_check(policy_name, gpu_details_list)