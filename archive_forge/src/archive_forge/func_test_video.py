import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
@pytest.mark.parametrize('streaming', [True, False])
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
def test_video(streaming, ctx):
    rng = np.random.RandomState(0)
    shape = (256, 64, 64, 3)
    video_data = rng.randint(0, 256, size=np.prod(shape), dtype=np.uint8).reshape(shape)
    with ctx() as path:
        with bf.BlobFile(path, mode='wb', streaming=streaming) as wf:
            with imageio.get_writer(wf, format='ffmpeg', quality=None, codec='libx264rgb', pixelformat='bgr24', output_params=['-f', 'mp4', '-crf', '0']) as w:
                for frame in video_data:
                    w.append_data(frame)
        with bf.BlobFile(path, mode='rb', streaming=streaming) as rf:
            with imageio.get_reader(rf, format='ffmpeg', input_params=['-f', 'mp4']) as r:
                for idx, frame in enumerate(r):
                    assert np.array_equal(frame, video_data[idx])
        with bf.BlobFile(path, mode='rb', streaming=streaming) as rf:
            container = av.open(rf)
            stream = container.streams.video[0]
            for idx, frame in enumerate(container.decode(stream)):
                assert np.array_equal(frame.to_image(), video_data[idx])