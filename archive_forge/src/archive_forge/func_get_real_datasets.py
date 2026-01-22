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
def get_real_datasets():
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    tmpdir = tempfile.TemporaryDirectory()
    test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url, root=tmpdir.name))
    tokenizer = get_tokenizer('basic_english')

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding='utf8'))))
    train_dataset = data_process(iter(io.open(train_filepath, encoding='utf8')))
    valid_dataset = data_process(iter(io.open(valid_filepath, encoding='utf8')))
    test_dataset = data_process(iter(io.open(test_filepath, encoding='utf8')))
    return DatasetsInfo(len(vocab.stoi), train_dataset, valid_dataset, test_dataset)