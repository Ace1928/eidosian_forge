import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def save_disconnects(self):
    """
        Saves the local list of disconnects to file.
        """
    file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(self.disconnects, f, pickle.HIGHEST_PROTOCOL)