import os
from parlai.core import build_data
from parlai.core.opt import Opt
def get_style_labeled_data_folder(datapath: str) -> str:
    return os.path.join(datapath, TASK_FOLDER_NAME, 'labeled_datasets')