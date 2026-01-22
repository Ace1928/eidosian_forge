import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
def deepspeed_launcher(args):
    import torch.distributed.run as distrib_run
    if not is_deepspeed_available():
        raise ImportError('DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source.')
    cmd, current_env = prepare_deepspeed_cmd_env(args)
    if not check_cuda_p2p_ib_support():
        message = "Using RTX 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled."
        warn = False
        if 'NCCL_P2P_DISABLE' not in current_env:
            current_env['NCCL_P2P_DISABLE'] = '1'
            warn = True
        if 'NCCL_IB_DISABLE' not in current_env:
            current_env['NCCL_IB_DISABLE'] = '1'
            warn = True
        if warn:
            logger.warning(message)
    if args.num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        with open('.deepspeed_env', 'a') as f:
            for key, value in current_env.items():
                if ';' in value or ' ' in value:
                    continue
                f.write(f'{key}={value}\n')
        process = subprocess.Popen(cmd, env=current_env)
        process.wait()
        if process.returncode != 0:
            if not args.quiet:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
            else:
                sys.exit(1)
    else:
        debug = getattr(args, 'debug', False)
        args = _filter_args(args, distrib_run.get_args_parser(), ['--training_script', args.training_script, '--training_script_args', args.training_script_args])
        with patch_environment(**current_env):
            try:
                distrib_run.run(args)
            except Exception:
                if is_rich_available() and debug:
                    console = get_console()
                    console.print('\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]')
                    console.print_exception(suppress=[__file__], show_locals=False)
                else:
                    raise