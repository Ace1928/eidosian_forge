import logging
from pathlib import Path
import shutil
import tempfile
from torchvision.datasets import MNIST
def setup_cached_mnist():
    done, tentatives = (False, 0)
    while not done and tentatives < 5:
        MNIST.mirrors = ['https://github.com/blefaudeux/mnist_dataset/raw/main/'] + MNIST.mirrors
        try:
            _ = MNIST(transform=None, download=True, root=TEMPDIR)
            done = True
        except RuntimeError as e:
            logging.warning(e)
            mnist_root = Path(TEMPDIR + '/MNIST')
            shutil.rmtree(str(mnist_root))
        tentatives += 1
    if done is False:
        logging.error('Could not download MNIST dataset')
        exit(-1)
    else:
        logging.info('Dataset downloaded')