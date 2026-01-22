import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def handle_bad_disconnect(self, worker_id):
    """
        Update the number of bad disconnects for the given worker, block them if they've
        exceeded the disconnect limit.
        """
    if not self.is_sandbox:
        self.mturk_workers[worker_id].disconnects += 1
        self.disconnects.append({'time': time.time(), 'id': worker_id})
        if self.mturk_workers[worker_id].disconnects > MAX_DISCONNECTS:
            if self.opt['hard_block']:
                text = "This worker has repeatedly disconnected from these tasks, which require constant connection to complete properly as they involve interaction with other Turkers. They have been blocked after being warned and failing to adhere. This was done in order to ensure a better experience for other workers who don't disconnect."
                self.mturk_manager.block_worker(worker_id, text)
                shared_utils.print_and_log(logging.INFO, 'Worker {} blocked - too many disconnects'.format(worker_id), True)
            elif self.opt['disconnect_qualification'] is not None:
                self.mturk_manager.soft_block_worker(worker_id, 'disconnect_qualification')
                shared_utils.print_and_log(logging.INFO, 'Worker {} soft blocked - too many disconnects'.format(worker_id), True)