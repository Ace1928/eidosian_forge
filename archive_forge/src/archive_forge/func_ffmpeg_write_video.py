import os
import subprocess as sp
import numpy as np
from proglog import proglog
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting
def ffmpeg_write_video(clip, filename, fps, codec='libx264', bitrate=None, preset='medium', withmask=False, write_logfile=False, audiofile=None, verbose=True, threads=None, ffmpeg_params=None, logger='bar'):
    """ Write the clip to a videofile. See VideoClip.write_videofile for details
    on the parameters.
    """
    logger = proglog.default_bar_logger(logger)
    if write_logfile:
        logfile = open(filename + '.log', 'w+')
    else:
        logfile = None
    logger(message='Moviepy - Writing video %s\n' % filename)
    with FFMPEG_VideoWriter(filename, clip.size, fps, codec=codec, preset=preset, bitrate=bitrate, logfile=logfile, audiofile=audiofile, threads=threads, ffmpeg_params=ffmpeg_params) as writer:
        nframes = int(clip.duration * fps)
        for t, frame in clip.iter_frames(logger=logger, with_times=True, fps=fps, dtype='uint8'):
            if withmask:
                mask = 255 * clip.mask.get_frame(t)
                if mask.dtype != 'uint8':
                    mask = mask.astype('uint8')
                frame = np.dstack([frame, mask])
            writer.write_frame(frame)
    if write_logfile:
        logfile.close()
    logger(message='Moviepy - Done !')