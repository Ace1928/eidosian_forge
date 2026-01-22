import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def vad(waveform: Tensor, sample_rate: int, trigger_level: float=7.0, trigger_time: float=0.25, search_time: float=1.0, allowed_gap: float=0.25, pre_trigger_time: float=0.0, boot_time: float=0.35, noise_up_time: float=0.1, noise_down_time: float=0.01, noise_reduction_amount: float=1.35, measure_freq: float=20.0, measure_duration: Optional[float]=None, measure_smooth_time: float=0.4, hp_filter_freq: float=50.0, lp_filter_freq: float=6000.0, hp_lifter_freq: float=150.0, lp_lifter_freq: float=2000.0) -> Tensor:
    """Voice Activity Detector. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Attempts to trim silence and quiet background sounds from the ends of recordings of speech.
    The algorithm currently uses a simple cepstral power measurement to detect voice,
    so may be fooled by other things, especially music.

    The effect can trim only from the front of the audio,
    so in order to trim from the back, the reverse effect must also be used.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(channels, time)` or `(time)`
            Tensor of shape `(channels, time)` is treated as a multi-channel recording
            of the same event and the resulting output will be trimmed to the earliest
            voice activity in any channel.
        sample_rate (int): Sample rate of audio signal.
        trigger_level (float, optional): The measurement level used to trigger activity detection.
            This may need to be cahnged depending on the noise level, signal level,
            and other characteristics of the input audio. (Default: 7.0)
        trigger_time (float, optional): The time constant (in seconds)
            used to help ignore short bursts of sound. (Default: 0.25)
        search_time (float, optional): The amount of audio (in seconds)
            to search for quieter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 1.0)
        allowed_gap (float, optional): The allowed gap (in seconds) between
            quieter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 0.25)
        pre_trigger_time (float, optional): The amount of audio (in seconds) to preserve
            before the trigger point and any found quieter/shorter bursts. (Default: 0.0)
        boot_time (float, optional) The algorithm (internally) uses adaptive noise
            estimation/reduction in order to detect the start of the wanted audio.
            This option sets the time for the initial noise estimate. (Default: 0.35)
        noise_up_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is increasing. (Default: 0.1)
        noise_down_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is decreasing. (Default: 0.01)
        noise_reduction_amount (float, optional) Amount of noise reduction to use in
            the detection algorithm (e.g. 0, 0.5, ...). (Default: 1.35)
        measure_freq (float, optional) Frequency of the algorithm's
            processing/measurements. (Default: 20.0)
        measure_duration: (float, optional) Measurement duration.
            (Default: Twice the measurement period; i.e. with overlap.)
        measure_smooth_time (float, optional) Time constant used to smooth
            spectral measurements. (Default: 0.4)
        hp_filter_freq (float, optional) "Brick-wall" frequency of high-pass filter applied
            at the input to the detector algorithm. (Default: 50.0)
        lp_filter_freq (float, optional) "Brick-wall" frequency of low-pass filter applied
            at the input to the detector algorithm. (Default: 6000.0)
        hp_lifter_freq (float, optional) "Brick-wall" frequency of high-pass lifter used
            in the detector algorithm. (Default: 150.0)
        lp_lifter_freq (float, optional) "Brick-wall" frequency of low-pass lifter used
            in the detector algorithm. (Default: 2000.0)

    Returns:
        Tensor: Tensor of audio of dimension `(..., time)`.

    Reference:
        - http://sox.sourceforge.net/sox.html
    """
    device = waveform.device
    if waveform.ndim > 2:
        warnings.warn(f'Expected input tensor dimension of 1 for single channel or 2 for multi-channel. Got {waveform.ndim} instead. Batch semantics is not supported. Please refer to https://github.com/pytorch/audio/issues/1348 and https://github.com/pytorch/audio/issues/1468.')
    measure_duration: float = 2.0 / measure_freq if measure_duration is None else measure_duration
    measure_len_ws = int(sample_rate * measure_duration + 0.5)
    measure_len_ns = measure_len_ws
    dft_len_ws = 16
    while dft_len_ws < measure_len_ws:
        dft_len_ws *= 2
    measure_period_ns = int(sample_rate / measure_freq + 0.5)
    measures_len = math.ceil(search_time * measure_freq)
    search_pre_trigger_len_ns = measures_len * measure_period_ns
    gap_len = int(allowed_gap * measure_freq + 0.5)
    fixed_pre_trigger_len_ns = int(pre_trigger_time * sample_rate + 0.5)
    samplesLen_ns = fixed_pre_trigger_len_ns + search_pre_trigger_len_ns + measure_len_ns
    spectrum_window = torch.zeros(measure_len_ws, device=device)
    for i in range(measure_len_ws):
        spectrum_window[i] = 2.0 / math.sqrt(float(measure_len_ws))
    spectrum_window *= torch.hann_window(measure_len_ws, device=device, dtype=torch.float)
    spectrum_start: int = int(hp_filter_freq / sample_rate * dft_len_ws + 0.5)
    spectrum_start: int = max(spectrum_start, 1)
    spectrum_end: int = int(lp_filter_freq / sample_rate * dft_len_ws + 0.5)
    spectrum_end: int = min(spectrum_end, dft_len_ws // 2)
    cepstrum_window = torch.zeros(spectrum_end - spectrum_start, device=device)
    for i in range(spectrum_end - spectrum_start):
        cepstrum_window[i] = 2.0 / math.sqrt(float(spectrum_end) - spectrum_start)
    cepstrum_window *= torch.hann_window(spectrum_end - spectrum_start, device=device, dtype=torch.float)
    cepstrum_start = math.ceil(sample_rate * 0.5 / lp_lifter_freq)
    cepstrum_end = math.floor(sample_rate * 0.5 / hp_lifter_freq)
    cepstrum_end = min(cepstrum_end, dft_len_ws // 4)
    if cepstrum_end <= cepstrum_start:
        raise ValueError(f'Expected cepstrum_start to be smaller than cepstrum_end.Found: cepstrum_start: {cepstrum_start}, cepstrum_end: {cepstrum_end}.')
    noise_up_time_mult = torch.tensor(math.exp(-1.0 / (noise_up_time * measure_freq)), device=device)
    noise_down_time_mult = torch.tensor(math.exp(-1.0 / (noise_down_time * measure_freq)), device=device)
    measure_smooth_time_mult = math.exp(-1.0 / (measure_smooth_time * measure_freq))
    trigger_meas_time_mult = math.exp(-1.0 / (trigger_time * measure_freq))
    boot_count_max = int(boot_time * measure_freq - 0.5)
    boot_count = measures_index = flushedLen_ns = 0
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    n_channels, ilen = waveform.size()
    mean_meas = torch.zeros(n_channels, device=device)
    spectrum = torch.zeros(n_channels, dft_len_ws, device=device)
    noise_spectrum = torch.zeros(n_channels, dft_len_ws, device=device)
    measures = torch.zeros(n_channels, measures_len, device=device)
    has_triggered: bool = False
    num_measures_to_flush: int = 0
    pos = 0
    for pos in range(measure_len_ns, ilen, measure_period_ns):
        for i in range(n_channels):
            meas: float = _measure(measure_len_ws=measure_len_ws, samples=waveform[i, pos - measure_len_ws:pos], spectrum=spectrum[i], noise_spectrum=noise_spectrum[i], spectrum_window=spectrum_window, spectrum_start=spectrum_start, spectrum_end=spectrum_end, cepstrum_window=cepstrum_window, cepstrum_start=cepstrum_start, cepstrum_end=cepstrum_end, noise_reduction_amount=noise_reduction_amount, measure_smooth_time_mult=measure_smooth_time_mult, noise_up_time_mult=noise_up_time_mult, noise_down_time_mult=noise_down_time_mult, boot_count=boot_count)
            measures[i, measures_index] = meas
            mean_meas[i] = mean_meas[i] * trigger_meas_time_mult + meas * (1.0 - trigger_meas_time_mult)
            has_triggered = has_triggered or mean_meas[i] >= trigger_level
            if has_triggered:
                n: int = measures_len
                k: int = measures_index
                jTrigger: int = n
                jZero: int = n
                j: int = 0
                for j in range(n):
                    if measures[i, k] >= trigger_level and j <= jTrigger + gap_len:
                        jZero = jTrigger = j
                    elif measures[i, k] == 0 and jTrigger >= jZero:
                        jZero = j
                    k = (k + n - 1) % n
                j = min(j, jZero)
                num_measures_to_flush = min(max(num_measures_to_flush, j), n)
        measures_index += 1
        measures_index = measures_index % measures_len
        if boot_count >= 0:
            boot_count = -1 if boot_count == boot_count_max else boot_count + 1
        if has_triggered:
            flushedLen_ns = (measures_len - num_measures_to_flush) * measure_period_ns
            break
    if not has_triggered:
        return waveform[..., :0].view(shape[:-1] + torch.Size([0]))
    res = waveform[:, pos - samplesLen_ns + flushedLen_ns:]
    return res.view(shape[:-1] + res.shape[-1:])