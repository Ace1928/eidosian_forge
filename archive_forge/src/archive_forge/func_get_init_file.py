import argparse
import os
import uuid
from pathlib import Path
import submitit
from xformers.benchmarks.LRA.run_tasks import benchmark, get_arg_parser
def get_init_file():
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f'{uuid.uuid4().hex}_init'
    if init_file.exists():
        os.remove(str(init_file))
    return init_file