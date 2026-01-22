import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
import numpy as np
import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.utils import make_table
def load_tasks(args):
    tasks.initialize_tasks()
    if args.open_llm_leaderboard_tasks:
        current_dir = os.getcwd()
        config_dir = os.path.join(current_dir, 'open_llm_leaderboard')
        lm_eval.tasks.include_path(config_dir)
        return ['arc_challenge_25_shot', 'hellaswag_10_shot', 'truthfulqa_mc2', 'winogrande_5_shot', 'gsm8k', 'mmlu']
    return args.tasks.split(',') if args.tasks else []