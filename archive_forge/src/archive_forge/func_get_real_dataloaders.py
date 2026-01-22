from collections import namedtuple
from distutils.version import LooseVersion
import io
import operator
import tempfile
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
def get_real_dataloaders(args, benchmark_config, model_specs, num_replicas=1, rank=0):
    """Return real dataloaders for training, testing and validation."""
    dataset_info = get_real_datasets()
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(dataset_info, benchmark_config, model_specs, num_replicas, rank)
    return (dataset_info.ntokens, train_dataloader, valid_dataloder, test_dataloader)