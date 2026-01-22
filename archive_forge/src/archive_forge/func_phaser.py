import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def phaser(waveform: Tensor, sample_rate: int, gain_in: float=0.4, gain_out: float=0.74, delay_ms: float=3.0, decay: float=0.4, mod_speed: float=0.5, sinusoidal: bool=True) -> Tensor:
    """Apply a phasing effect to the audio. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain_in (float, optional): desired input gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 1
        gain_out (float, optional): desired output gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 1e9
        delay_ms (float, optional): desired delay in milliseconds
            Allowed range of values are 0 to 5.0
        decay (float, optional):  desired decay relative to gain-in
            Allowed range of values are 0 to 0.99
        mod_speed (float, optional):  modulation speed in Hz
            Allowed range of values are 0.1 to 2
        sinusoidal (bool, optional):  If ``True``, uses sinusoidal modulation (preferable for multiple instruments)
            If ``False``, uses triangular modulation (gives single instruments a sharper phasing effect)
            (Default: ``True``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - Scott Lehman, `Effects Explained`_.

    .. _Effects Explained:
        https://web.archive.org/web/20051125072557/http://www.harmony-central.com/Effects/effects-explained.html
    """
    actual_shape = waveform.shape
    device, dtype = (waveform.device, waveform.dtype)
    waveform = waveform.view(-1, actual_shape[-1])
    delay_buf_len = int(delay_ms * 0.001 * sample_rate + 0.5)
    delay_buf = torch.zeros(waveform.shape[0], delay_buf_len, dtype=dtype, device=device)
    mod_buf_len = int(sample_rate / mod_speed + 0.5)
    if sinusoidal:
        wave_type = 'SINE'
    else:
        wave_type = 'TRIANGLE'
    mod_buf = _generate_wave_table(wave_type=wave_type, data_type='INT', table_size=mod_buf_len, min=1.0, max=float(delay_buf_len), phase=math.pi / 2, device=device)
    delay_pos = 0
    mod_pos = 0
    output_waveform_pre_gain_list = []
    waveform = waveform * gain_in
    delay_buf = delay_buf * decay
    waveform_list = [waveform[:, i] for i in range(waveform.size(1))]
    delay_buf_list = [delay_buf[:, i] for i in range(delay_buf.size(1))]
    mod_buf_list = [mod_buf[i] for i in range(mod_buf.size(0))]
    for i in range(waveform.shape[-1]):
        idx = int((delay_pos + mod_buf_list[mod_pos]) % delay_buf_len)
        mod_pos = (mod_pos + 1) % mod_buf_len
        delay_pos = (delay_pos + 1) % delay_buf_len
        temp = waveform_list[i] + delay_buf_list[idx]
        delay_buf_list[delay_pos] = temp * decay
        output_waveform_pre_gain_list.append(temp)
    output_waveform = torch.stack(output_waveform_pre_gain_list, dim=1).to(dtype=dtype, device=device)
    output_waveform.mul_(gain_out)
    return output_waveform.clamp(min=-1, max=1).view(actual_shape)