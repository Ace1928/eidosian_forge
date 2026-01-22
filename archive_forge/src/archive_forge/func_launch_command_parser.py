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
def launch_command_parser(subparsers=None):
    description = 'Launch a python script in a distributed scenario. Arguments can be passed in with either hyphens (`--num-processes=2`) or underscores (`--num_processes=2`)'
    if subparsers is not None:
        parser = subparsers.add_parser('launch', description=description, add_help=False, allow_abbrev=False, formatter_class=CustomHelpFormatter)
    else:
        parser = CustomArgumentParser('Accelerate launch command', description=description, add_help=False, allow_abbrev=False, formatter_class=CustomHelpFormatter)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('--config_file', default=None, help='The config file to use for the default values in the launching script.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Silence subprocess errors from the launch stack trace and only show the relevant tracebacks. (Only applicable to DeepSpeed and single-process configurations)')
    hardware_args = parser.add_argument_group('Hardware Selection Arguments', 'Arguments for selecting the hardware to be used.')
    hardware_args.add_argument('--cpu', default=False, action='store_true', help='Whether or not to force the training on the CPU.')
    hardware_args.add_argument('--multi_gpu', default=False, action='store_true', help='Whether or not this should launch a distributed GPU training.')
    hardware_args.add_argument('--tpu', default=False, action='store_true', help='Whether or not this should launch a TPU training.')
    hardware_args.add_argument('--ipex', default=False, action='store_true', help='Whether or not this should launch a Intel PyTorch Extension (IPEX) training.')
    resource_args = parser.add_argument_group('Resource Selection Arguments', 'Arguments for fine-tuning how available hardware should be used.')
    resource_args.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16', 'fp8'], help='Whether or not to use mixed precision training. Choose between FP16 and BF16 (bfloat16) training. BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.')
    resource_args.add_argument('--num_processes', type=int, default=None, help='The total number of processes to be launched in parallel.')
    resource_args.add_argument('--num_machines', type=int, default=None, help='The total number of machines used in this training.')
    resource_args.add_argument('--num_cpu_threads_per_process', type=int, default=None, help='The number of CPU threads per process. Can be tuned for optimal performance.')
    resource_args.add_argument('--dynamo_backend', type=str, choices=['no'] + [b.lower() for b in DYNAMO_BACKENDS], help='Choose a backend to optimize your training with dynamo, see more at https://github.com/pytorch/torchdynamo.')
    resource_args.add_argument('--dynamo_mode', type=str, default='default', choices=TORCH_DYNAMO_MODES, help='Choose a mode to optimize your training with dynamo.')
    resource_args.add_argument('--dynamo_use_fullgraph', default=False, action='store_true', help='Whether to use full graph mode for dynamo or it is ok to break model into several subgraphs')
    resource_args.add_argument('--dynamo_use_dynamic', default=False, action='store_true', help='Whether to enable dynamic shape tracing.')
    paradigm_args = parser.add_argument_group('Training Paradigm Arguments', 'Arguments for selecting which training paradigm to be used.')
    paradigm_args.add_argument('--use_deepspeed', default=False, action='store_true', help='Whether to use deepspeed.')
    paradigm_args.add_argument('--use_fsdp', default=False, action='store_true', help='Whether to use fsdp.')
    paradigm_args.add_argument('--use_megatron_lm', default=False, action='store_true', help='Whether to use Megatron-LM.')
    paradigm_args.add_argument('--use_xpu', default=False, action='store_true', help='Whether to use IPEX plugin to speed up training on XPU specifically.')
    distributed_args = parser.add_argument_group('Distributed GPUs', 'Arguments related to distributed GPU training.')
    distributed_args.add_argument('--gpu_ids', default=None, help='What GPUs (by id) should be used for training on this machine as a comma-seperated list')
    distributed_args.add_argument('--same_network', default=False, action='store_true', help='Whether all machines used for multinode training exist on the same local network.')
    distributed_args.add_argument('--machine_rank', type=int, default=None, help='The rank of the machine on which this script is launched.')
    distributed_args.add_argument('--main_process_ip', type=str, default=None, help='The IP address of the machine of rank 0.')
    distributed_args.add_argument('--main_process_port', type=int, default=None, help='The port to use to communicate with the machine of rank 0.')
    distributed_args.add_argument('-t', '--tee', default='0', type=str, help='Tee std streams into a log file and also to console.')
    distributed_args.add_argument('--role', type=str, default='default', help='User-defined role for the workers.')
    distributed_args.add_argument('--rdzv_backend', type=str, default='static', help="The rendezvous method to use, such as 'static' (the default) or 'c10d'")
    distributed_args.add_argument('--rdzv_conf', type=str, default='', help='Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).')
    distributed_args.add_argument('--max_restarts', type=int, default=0, help='Maximum number of worker group restarts before failing.')
    distributed_args.add_argument('--monitor_interval', type=float, default=5, help='Interval, in seconds, to monitor the state of workers.')
    parser.add_argument('-m', '--module', action='store_true', help="Change each process to interpret the launch script as a Python module, executing with the same behavior as 'python -m'.")
    parser.add_argument('--no_python', action='store_true', help="Skip prepending the training script with 'python' - just execute it directly. Useful when the script is not a Python script.")
    tpu_args = parser.add_argument_group('TPU', 'Arguments related to TPU.')
    tpu_args.add_argument('--tpu_cluster', action='store_true', dest='tpu_use_cluster', help='Whether to use a GCP TPU pod for training.')
    tpu_args.add_argument('--no_tpu_cluster', action='store_false', dest='tpu_use_cluster', help='Should not be passed explicitly, this is for internal use only.')
    tpu_args.add_argument('--tpu_use_sudo', action='store_true', help='Whether to use `sudo` when running the TPU training script in each pod.')
    tpu_args.add_argument('--vm', type=str, action='append', help='List of single Compute VM instance names. If not provided we assume usage of instance groups. For TPU pods.')
    tpu_args.add_argument('--env', type=str, action='append', help='List of environment variables to set on the Compute VM instances. For TPU pods.')
    tpu_args.add_argument('--main_training_function', type=str, default=None, help='The name of the main function to be executed in your script (only for TPU training).')
    tpu_args.add_argument('--downcast_bf16', action='store_true', help='Whether when using bf16 precision on TPUs if both float and double tensors are cast to bfloat16 or if double tensors remain as float32.')
    deepspeed_args = parser.add_argument_group('DeepSpeed Arguments', 'Arguments related to DeepSpeed.')
    deepspeed_args.add_argument('--deepspeed_config_file', default=None, type=str, help='DeepSpeed config file.')
    deepspeed_args.add_argument('--zero_stage', default=None, type=int, help="DeepSpeed's ZeRO optimization stage (useful only when `use_deepspeed` flag is passed). If unspecified, will default to `2`.")
    deepspeed_args.add_argument('--offload_optimizer_device', default=None, type=str, help="Decides where (none|cpu|nvme) to offload optimizer states (useful only when `use_deepspeed` flag is passed). If unspecified, will default to 'none'.")
    deepspeed_args.add_argument('--offload_param_device', default=None, type=str, help="Decides where (none|cpu|nvme) to offload parameters (useful only when `use_deepspeed` flag is passed). If unspecified, will default to 'none'.")
    deepspeed_args.add_argument('--offload_optimizer_nvme_path', default=None, type=str, help="Decides Nvme Path to offload optimizer states (useful only when `use_deepspeed` flag is passed). If unspecified, will default to 'none'.")
    deepspeed_args.add_argument('--offload_param_nvme_path', default=None, type=str, help="Decides Nvme Path to offload parameters (useful only when `use_deepspeed` flag is passed). If unspecified, will default to 'none'.")
    deepspeed_args.add_argument('--gradient_accumulation_steps', default=None, type=int, help='No of gradient_accumulation_steps used in your training script (useful only when `use_deepspeed` flag is passed). If unspecified, will default to `1`.')
    deepspeed_args.add_argument('--gradient_clipping', default=None, type=float, help='gradient clipping value used in your training script (useful only when `use_deepspeed` flag is passed). If unspecified, will default to `1.0`.')
    deepspeed_args.add_argument('--zero3_init_flag', default=None, type=str, help='Decides Whether (true|false) to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with DeepSpeed ZeRO Stage-3. If unspecified, will default to `true`.')
    deepspeed_args.add_argument('--zero3_save_16bit_model', default=None, type=str, help='Decides Whether (true|false) to save 16-bit model weights when using ZeRO Stage-3. Only applicable with DeepSpeed ZeRO Stage-3. If unspecified, will default to `false`.')
    deepspeed_args.add_argument('--deepspeed_hostfile', default=None, type=str, help='DeepSpeed hostfile for configuring multi-node compute resources.')
    deepspeed_args.add_argument('--deepspeed_exclusion_filter', default=None, type=str, help='DeepSpeed exclusion filter string when using mutli-node setup.')
    deepspeed_args.add_argument('--deepspeed_inclusion_filter', default=None, type=str, help='DeepSpeed inclusion filter string when using mutli-node setup.')
    deepspeed_args.add_argument('--deepspeed_multinode_launcher', default=None, type=str, help='DeepSpeed multi-node launcher to use. If unspecified, will default to `pdsh`.')
    fsdp_args = parser.add_argument_group('FSDP Arguments', 'Arguments related to Fully Shared Data Parallelism.')
    fsdp_args.add_argument('--fsdp_offload_params', default='false', type=str, help='Decides Whether (true|false) to offload parameters and gradients to CPU. (useful only when `use_fsdp` flag is passed).')
    fsdp_args.add_argument('--fsdp_min_num_params', type=int, default=100000000.0, help="FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `use_fsdp` flag is passed).")
    fsdp_args.add_argument('--fsdp_sharding_strategy', type=str, default='FULL_SHARD', help="FSDP's Sharding Strategy. (useful only when `use_fsdp` flag is passed).")
    fsdp_args.add_argument('--fsdp_auto_wrap_policy', type=str, default=None, help="FSDP's auto wrap policy. (useful only when `use_fsdp` flag is passed).")
    fsdp_args.add_argument('--fsdp_transformer_layer_cls_to_wrap', default=None, type=str, help='Transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `use_fsdp` flag is passed).')
    fsdp_args.add_argument('--fsdp_backward_prefetch_policy', default=None, type=str, help='This argument is deprecated and will be removed in version 0.27.0 of 🤗 Accelerate. Use `fsdp_backward_prefetch` instead.')
    fsdp_args.add_argument('--fsdp_backward_prefetch', default=None, type=str, help="FSDP's backward prefetch policy. (useful only when `use_fsdp` flag is passed).")
    fsdp_args.add_argument('--fsdp_state_dict_type', default=None, type=str, help="FSDP's state dict type. (useful only when `use_fsdp` flag is passed).")
    fsdp_args.add_argument('--fsdp_forward_prefetch', default='false', type=str, help='If True, then FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass (useful only when `use_fsdp` flag is passed).')
    fsdp_args.add_argument('--fsdp_use_orig_params', default='true', type=str, help='If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres. (useful only when `use_fsdp` flag is passed).')
    fsdp_args.add_argument('--fsdp_cpu_ram_efficient_loading', default='true', type=str, help='If True, only the first process loads the pretrained model checkoint while all other processes have empty weights. Only applicable for 🤗 Transformers. When using this, `--fsdp_sync_module_states` needs to True. (useful only when `use_fsdp` flag is passed).')
    fsdp_args.add_argument('--fsdp_sync_module_states', default='true', type=str, help='If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0. (useful only when `use_fsdp` flag is passed).')
    megatron_lm_args = parser.add_argument_group('Megatron-LM Arguments', 'Arguments related to Megatron-LM.')
    megatron_lm_args.add_argument('--megatron_lm_tp_degree', type=int, default=1, help="Megatron-LM's Tensor Parallelism (TP) degree. (useful only when `use_megatron_lm` flag is passed).")
    megatron_lm_args.add_argument('--megatron_lm_pp_degree', type=int, default=1, help="Megatron-LM's Pipeline Parallelism (PP) degree. (useful only when `use_megatron_lm` flag is passed).")
    megatron_lm_args.add_argument('--megatron_lm_num_micro_batches', type=int, default=None, help="Megatron-LM's number of micro batches when PP degree > 1. (useful only when `use_megatron_lm` flag is passed).")
    megatron_lm_args.add_argument('--megatron_lm_sequence_parallelism', default=None, type=str, help='Decides Whether (true|false) to enable Sequence Parallelism when TP degree > 1. (useful only when `use_megatron_lm` flag is passed).')
    megatron_lm_args.add_argument('--megatron_lm_recompute_activations', default=None, type=str, help='Decides Whether (true|false) to enable Selective Activation Recomputation. (useful only when `use_megatron_lm` flag is passed).')
    megatron_lm_args.add_argument('--megatron_lm_use_distributed_optimizer', default=None, type=str, help='Decides Whether (true|false) to use distributed optimizer which shards optimizer state and gradients across Data Pralellel (DP) ranks. (useful only when `use_megatron_lm` flag is passed).')
    megatron_lm_args.add_argument('--megatron_lm_gradient_clipping', default=1.0, type=float, help="Megatron-LM's gradient clipping value based on global L2 Norm (0 to disable). (useful only when `use_megatron_lm` flag is passed).")
    aws_args = parser.add_argument_group('AWS Arguments', 'Arguments related to AWS.')
    aws_args.add_argument('--aws_access_key_id', type=str, default=None, help='The AWS_ACCESS_KEY_ID used to launch the Amazon SageMaker training job')
    aws_args.add_argument('--aws_secret_access_key', type=str, default=None, help='The AWS_SECRET_ACCESS_KEY used to launch the Amazon SageMaker training job.')
    parser.add_argument('--debug', action='store_true', help='Whether to print out the torch.distributed stack trace when something fails.')
    parser.add_argument('training_script', type=str, help='The full path to the script to be launched in parallel, followed by all the arguments for the training script.')
    mpirun_args = parser.add_argument_group('MPI Arguments', 'Arguments related to mpirun for Multi-CPU')
    mpirun_args.add_argument('--mpirun_hostfile', type=str, default=None, help='Location for a hostfile for using Accelerate to launch a multi-CPU training job with mpirun. This will get passed to the MPI --hostfile or -f parameter, depending on which MPI program is installed.')
    mpirun_args.add_argument('--mpirun_ccl', type=int, default=1, help='The number of oneCCL worker threads when using Accelerate to launch multi-CPU training with mpirun.')
    parser.add_argument('training_script_args', nargs=argparse.REMAINDER, help='Arguments of the training script.')
    if subparsers is not None:
        parser.set_defaults(func=launch_command)
    return parser