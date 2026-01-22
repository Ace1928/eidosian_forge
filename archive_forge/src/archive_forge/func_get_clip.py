import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
def get_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], int]:
    """
        Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
    if idx >= self.num_clips():
        raise IndexError(f'Index {idx} out of range ({self.num_clips()} number of clips)')
    video_idx, clip_idx = self.get_clip_location(idx)
    video_path = self.video_paths[video_idx]
    clip_pts = self.clips[video_idx][clip_idx]
    from torchvision import get_video_backend
    backend = get_video_backend()
    if backend == 'pyav':
        if self._video_width != 0:
            raise ValueError("pyav backend doesn't support _video_width != 0")
        if self._video_height != 0:
            raise ValueError("pyav backend doesn't support _video_height != 0")
        if self._video_min_dimension != 0:
            raise ValueError("pyav backend doesn't support _video_min_dimension != 0")
        if self._video_max_dimension != 0:
            raise ValueError("pyav backend doesn't support _video_max_dimension != 0")
        if self._audio_samples != 0:
            raise ValueError("pyav backend doesn't support _audio_samples != 0")
    if backend == 'pyav':
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        video, audio, info = read_video(video_path, start_pts, end_pts)
    else:
        _info = _probe_video_from_file(video_path)
        video_fps = _info.video_fps
        audio_fps = None
        video_start_pts = cast(int, clip_pts[0].item())
        video_end_pts = cast(int, clip_pts[-1].item())
        audio_start_pts, audio_end_pts = (0, -1)
        audio_timebase = Fraction(0, 1)
        video_timebase = Fraction(_info.video_timebase.numerator, _info.video_timebase.denominator)
        if _info.has_audio:
            audio_timebase = Fraction(_info.audio_timebase.numerator, _info.audio_timebase.denominator)
            audio_start_pts = pts_convert(video_start_pts, video_timebase, audio_timebase, math.floor)
            audio_end_pts = pts_convert(video_end_pts, video_timebase, audio_timebase, math.ceil)
            audio_fps = _info.audio_sample_rate
        video, audio, _ = _read_video_from_file(video_path, video_width=self._video_width, video_height=self._video_height, video_min_dimension=self._video_min_dimension, video_max_dimension=self._video_max_dimension, video_pts_range=(video_start_pts, video_end_pts), video_timebase=video_timebase, audio_samples=self._audio_samples, audio_channels=self._audio_channels, audio_pts_range=(audio_start_pts, audio_end_pts), audio_timebase=audio_timebase)
        info = {'video_fps': video_fps}
        if audio_fps is not None:
            info['audio_fps'] = audio_fps
    if self.frame_rate is not None:
        resampling_idx = self.resampling_idxs[video_idx][clip_idx]
        if isinstance(resampling_idx, torch.Tensor):
            resampling_idx = resampling_idx - resampling_idx[0]
        video = video[resampling_idx]
        info['video_fps'] = self.frame_rate
    assert len(video) == self.num_frames, f'{video.shape} x {self.num_frames}'
    if self.output_format == 'TCHW':
        video = video.permute(0, 3, 1, 2)
    return (video, audio, info, video_idx)