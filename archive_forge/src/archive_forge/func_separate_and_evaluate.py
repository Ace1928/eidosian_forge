import argparse
import functools
import json
import multiprocessing
from typing import Optional, Union
import musdb
import museval
import torch
import tqdm
from openunmix import utils
def separate_and_evaluate(track: musdb.MultiTrack, targets: list, model_str_or_path: str, niter: int, output_dir: str, eval_dir: str, residual: bool, mus, aggregate_dict: dict=None, device: Union[str, torch.device]='cpu', wiener_win_len: Optional[int]=None, filterbank='torch') -> str:
    separator = utils.load_separator(model_str_or_path=model_str_or_path, targets=targets, niter=niter, residual=residual, wiener_win_len=wiener_win_len, device=device, pretrained=True, filterbank=filterbank)
    separator.freeze()
    separator.to(device)
    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
    audio = utils.preprocess(audio, track.rate, separator.sample_rate)
    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)
    for key in estimates:
        estimates[key] = estimates[key][0].cpu().detach().numpy().T
    if output_dir:
        mus.save_estimates(estimates, track, output_dir)
    scores = museval.eval_mus_track(track, estimates, output_dir=eval_dir)
    return scores