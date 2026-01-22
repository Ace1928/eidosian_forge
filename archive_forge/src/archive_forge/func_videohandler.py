import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def videohandler(extension, data):
    if extension not in 'mp4 ogv mjpeg avi mov h264 mpg webm wmv'.split():
        return None
    try:
        import torchvision.io
    except ImportError as e:
        raise ModuleNotFoundError('Package `torchvision` is required to be installed for default video file loader.Please use `pip install torchvision` or `conda install torchvision -c pytorch`to install the package') from e
    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f'file.{extension}')
        with open(fname, 'wb') as stream:
            stream.write(data)
            return torchvision.io.read_video(fname)