from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.legacy_2018.mturk_manager import MTurkManager
import parlai.mturk.core.legacy_2018.mturk_utils as mturk_utils
from worlds import ControllableDialogEval, PersonasGenerator, PersonaAssignWorld
from task_config import task_config
import model_configs as mcf
from threading import Lock
import gc
import datetime
import json
import os
import sys
import copy
import random
import pprint
from parlai.utils.logging import ParlaiLogger, INFO
def run_onboard(worker):
    worker.personas_generator = personas_generator
    world = PersonaAssignWorld(start_opt, worker)
    world.parley()
    world.shutdown()