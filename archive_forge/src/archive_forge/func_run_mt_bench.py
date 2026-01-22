from typing import OrderedDict
import signal
import os
import json
import subprocess
import argparse
import time
import requests
import re
import coolname
def run_mt_bench(mt_bench_path, model_name):
    working_dir = os.path.join(mt_bench_path, 'fastchat', 'llm_judge')
    if os.path.exists(os.path.join(working_dir, 'data', 'mt_bench', 'model_answer', f'{model_name}.jsonl')):
        return
    commands = [f'python gen_api_answer.py --model {model_name} --max-tokens {MAX_CONTEXT} --parallel 128 --openai-api-base http://localhost:18888/v1', f'python gen_judgment.py --model-list {model_name} --parallel 8 --mode single']
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{env.get('PYTHONPATH', '')}:{mt_bench_path}'
    for command in commands:
        subprocess.run(command, shell=True, cwd=working_dir, env=env)