import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def should_run_distribute_coordinator(config):
    """Checks the config to see whether to run distribute coordinator."""
    if not hasattr(config, '_distribute_coordinator_mode') or config._distribute_coordinator_mode is None:
        logging.info('Not using Distribute Coordinator.')
        return False
    if not isinstance(config._distribute_coordinator_mode, six.string_types) or config._distribute_coordinator_mode not in [dc.CoordinatorMode.STANDALONE_CLIENT, dc.CoordinatorMode.INDEPENDENT_WORKER]:
        logging.warning('Unexpected distribute_coordinator_mode: %r', config._distribute_coordinator_mode)
        return False
    if not config.cluster_spec:
        logging.warning('Running `train_and_evaluate` locally, ignoring `experimental_distribute_coordinator_mode`.')
        return False
    return True