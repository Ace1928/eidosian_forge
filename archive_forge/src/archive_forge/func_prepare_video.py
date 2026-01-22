import argparse
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
from transformers.utils import logging
def prepare_video():
    from decord import VideoReader, cpu
    np.random.seed(0)

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.

        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.

        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    file_path = hf_hub_download(repo_id='nielsr/video-demo', filename='eating_spaghetti.mp4', repo_type='dataset')
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=6, frame_sample_rate=4, seg_len=len(videoreader))
    video = videoreader.get_batch(indices).asnumpy()
    return video