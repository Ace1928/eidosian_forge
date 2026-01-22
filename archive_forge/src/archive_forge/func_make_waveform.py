from __future__ import annotations
import ast
import csv
import inspect
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional
import numpy as np
import PIL
import PIL.Image
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import components, oauth, processing_utils, routes, utils, wasm_utils
from gradio.context import Context, LocalContext
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import EventData
from gradio.exceptions import Error
from gradio.flagging import CSVLogger
@document()
def make_waveform(audio: str | tuple[int, np.ndarray], *, bg_color: str='#f3f4f6', bg_image: str | None=None, fg_alpha: float=0.75, bars_color: str | tuple[str, str]=('#fbbf24', '#ea580c'), bar_count: int=50, bar_width: float=0.6, animate: bool=False) -> str:
    """
    Generates a waveform video from an audio file. Useful for creating an easy to share audio visualization. The output should be passed into a `gr.Video` component.
    Parameters:
        audio: Audio file path or tuple of (sample_rate, audio_data)
        bg_color: Background color of waveform (ignored if bg_image is provided)
        bg_image: Background image of waveform
        fg_alpha: Opacity of foreground waveform
        bars_color: Color of waveform bars. Can be a single color or a tuple of (start_color, end_color) of gradient
        bar_count: Number of bars in waveform
        bar_width: Width of bars in waveform. 1 represents full width, 0.5 represents half width, etc.
        animate: If true, the audio waveform overlay will be animated, if false, it will be static.
    Returns:
        A filepath to the output video in mp4 format.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if isinstance(audio, str):
        audio_file = audio
        audio = processing_utils.audio_from_file(audio)
    else:
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        processing_utils.audio_to_file(audio[0], audio[1], tmp_wav.name, format='wav')
        audio_file = tmp_wav.name
    if not os.path.isfile(audio_file):
        raise ValueError('Audio file not found.')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found.')
    duration = round(len(audio[1]) / audio[0], 4)

    def hex_to_rgb(hex_str):
        return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]

    def get_color_gradient(c1, c2, n):
        if n < 1:
            raise ValueError('Must have at least one stop in gradient')
        c1_rgb = np.array(hex_to_rgb(c1)) / 255
        c2_rgb = np.array(hex_to_rgb(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [(1 - mix) * c1_rgb + mix * c2_rgb for mix in mix_pcts]
        return ['#' + ''.join((f'{int(round(val * 255)):02x}' for val in item)) for item in rgb_colors]
    samples = audio[1]
    if len(samples.shape) > 1:
        samples = np.mean(samples, 1)
    bins_to_pad = bar_count - len(samples) % bar_count
    samples = np.pad(samples, [(0, bins_to_pad)])
    samples = np.reshape(samples, (bar_count, -1))
    samples = np.abs(samples)
    samples = np.max(samples, 1)
    with utils.MatplotlibBackendMananger():
        plt.clf()
        color = bars_color if isinstance(bars_color, str) else get_color_gradient(bars_color[0], bars_color[1], bar_count)
        if animate:
            fig = plt.figure(figsize=(5, 1), dpi=200, frameon=False)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.axis('off')
        plt.margins(x=0)
        bar_alpha = fg_alpha if animate else 1.0
        barcollection = plt.bar(np.arange(0, bar_count), samples * 2, bottom=-1 * samples, width=bar_width, color=color, alpha=bar_alpha)
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        savefig_kwargs: dict[str, Any] = {'bbox_inches': 'tight'}
        if bg_image is not None:
            savefig_kwargs['transparent'] = True
            if animate:
                savefig_kwargs['facecolor'] = 'none'
        else:
            savefig_kwargs['facecolor'] = bg_color
        plt.savefig(tmp_img.name, **savefig_kwargs)
        if not animate:
            waveform_img = PIL.Image.open(tmp_img.name)
            waveform_img = waveform_img.resize((1000, 400))
            if bg_image is not None:
                waveform_array = np.array(waveform_img)
                waveform_array[:, :, 3] = waveform_array[:, :, 3] * fg_alpha
                waveform_img = PIL.Image.fromarray(waveform_array)
                bg_img = PIL.Image.open(bg_image)
                waveform_width, waveform_height = waveform_img.size
                bg_width, bg_height = bg_img.size
                if waveform_width != bg_width:
                    bg_img = bg_img.resize((waveform_width, 2 * int(bg_height * waveform_width / bg_width / 2)))
                    bg_width, bg_height = bg_img.size
                composite_height = max(bg_height, waveform_height)
                composite = PIL.Image.new('RGBA', (waveform_width, composite_height), '#FFFFFF')
                composite.paste(bg_img, (0, composite_height - bg_height))
                composite.paste(waveform_img, (0, composite_height - waveform_height), waveform_img)
                composite.save(tmp_img.name)
                img_width, img_height = composite.size
            else:
                img_width, img_height = waveform_img.size
                waveform_img.save(tmp_img.name)
        else:

            def _animate(_):
                for idx, b in enumerate(barcollection):
                    rand_height = np.random.uniform(0.8, 1.2)
                    b.set_height(samples[idx] * rand_height * 2)
                    b.set_y((-rand_height * samples)[idx])
            frames = int(duration * 10)
            anim = FuncAnimation(fig, _animate, repeat=False, blit=False, frames=frames, interval=100)
            anim.save(tmp_img.name, writer='pillow', fps=10, codec='png', savefig_kwargs=savefig_kwargs)
    output_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if animate and bg_image is not None:
        ffmpeg_cmd = [ffmpeg, '-loop', '1', '-i', bg_image, '-i', tmp_img.name, '-i', audio_file, '-filter_complex', '[0:v]scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2[bg];[1:v]format=rgba,colorchannelmixer=aa=1.0[ov];[bg][ov]overlay=(main_w-overlay_w*0.9)/2:main_h-overlay_h*0.9/2[output]', '-t', str(duration), '-map', '[output]', '-map', '2:a', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', output_mp4.name]
    elif animate and bg_image is None:
        ffmpeg_cmd = [ffmpeg, '-i', tmp_img.name, '-i', audio_file, '-filter_complex', '[0:v][1:a]concat=n=1:v=1:a=1[v];[v]scale=1000:400,format=yuv420p[v_scaled]', '-map', '[v_scaled]', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', output_mp4.name]
    else:
        ffmpeg_cmd = [ffmpeg, '-loop', '1', '-i', tmp_img.name, '-i', audio_file, '-vf', f'color=c=#FFFFFF77:s={img_width}x{img_height}[bar];[0][bar]overlay=-w+(w/{duration})*t:H-h:shortest=1', '-t', str(duration), '-y', output_mp4.name]
    subprocess.check_call(ffmpeg_cmd)
    return output_mp4.name