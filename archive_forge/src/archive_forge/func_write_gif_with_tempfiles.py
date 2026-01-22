import os
import subprocess as sp
import numpy as np
import proglog
from moviepy.compat import DEVNULL
from moviepy.config import get_setting
from moviepy.decorators import requires_duration, use_clip_fps_by_default
from moviepy.tools import subprocess_call
@requires_duration
@use_clip_fps_by_default
def write_gif_with_tempfiles(clip, filename, fps=None, program='ImageMagick', opt='OptimizeTransparency', fuzz=1, verbose=True, loop=0, dispose=True, colors=None, logger='bar'):
    """ Write the VideoClip to a GIF file.


    Converts a VideoClip into an animated GIF using ImageMagick
    or ffmpeg. Does the same as write_gif (see this one for more
    docstring), but writes every frame to a file instead of passing
    them in the RAM. Useful on computers with little RAM.

    """
    logger = proglog.default_bar_logger(logger)
    fileName, ext = os.path.splitext(filename)
    tt = np.arange(0, clip.duration, 1.0 / fps)
    tempfiles = []
    logger(message='MoviePy - Building file %s\n' % filename)
    logger(message='MoviePy - - Generating GIF frames')
    for i, t in logger.iter_bar(t=list(enumerate(tt))):
        name = '%s_GIFTEMP%04d.png' % (fileName, i + 1)
        tempfiles.append(name)
        clip.save_frame(name, t, withmask=True)
    delay = int(100.0 / fps)
    if program == 'ImageMagick':
        logger(message='MoviePy - - Optimizing GIF with ImageMagick...')
        cmd = [get_setting('IMAGEMAGICK_BINARY'), '-delay', '%d' % delay, '-dispose', '%d' % (2 if dispose else 1), '-loop', '%d' % loop, '%s_GIFTEMP*.png' % fileName, '-coalesce', '-fuzz', '%02d' % fuzz + '%', '-layers', '%s' % opt] + (['-colors', '%d' % colors] if colors is not None else []) + [filename]
    elif program == 'ffmpeg':
        cmd = [get_setting('FFMPEG_BINARY'), '-y', '-f', 'image2', '-r', str(fps), '-i', fileName + '_GIFTEMP%04d.png', '-r', str(fps), filename]
    try:
        subprocess_call(cmd, logger=logger)
        logger(message='MoviePy - GIF ready: %s.' % filename)
    except (IOError, OSError) as err:
        error = 'MoviePy Error: creation of %s failed because of the following error:\n\n%s.\n\n.' % (filename, str(err))
        if program == 'ImageMagick':
            error = error + "This error can be due to the fact that ImageMagick is not installed on your computer, or (for Windows users) that you didn't specify the path to the ImageMagick binary in file config_defaults.py."
        raise IOError(error)
    for f in tempfiles:
        os.remove(f)