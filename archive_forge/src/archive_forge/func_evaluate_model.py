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
def evaluate_model(args):
    try:
        task_list = load_tasks(args)
        results = evaluator.simple_evaluate(model=args.model, model_args=args.model_args, tasks=task_list, num_fewshot=args.num_fewshot, batch_size=args.batch_size, max_batch_size=args.max_batch_size, device=args.device, use_cache=args.use_cache, limit=args.limit, decontamination_ngrams_path=args.decontamination_ngrams_path, check_integrity=args.check_integrity, write_out=args.write_out, log_samples=args.log_samples, gen_kwargs=args.gen_kwargs)
        handle_output(args, results, logger)
    except Exception as e:
        logger.error(f'An error occurred during evaluation: {e}')
        sys.exit(1)