from __future__ import division
import logging
import os
import re
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting  # ffmpeg, ffmpeg.exe, etc...
from moviepy.tools import cvsecs
def ffmpeg_parse_infos(filename, print_infos=False, check_duration=True, fps_source='tbr'):
    """Get file infos using ffmpeg.

    Returns a dictionnary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration", "audio_found", "audio_fps"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncomplete frames at the end, which raises an error.

    """
    is_GIF = filename.endswith('.gif')
    cmd = [get_setting('FFMPEG_BINARY'), '-i', filename]
    if is_GIF:
        cmd += ['-f', 'null', '/dev/null']
    popen_params = {'bufsize': 10 ** 5, 'stdout': sp.PIPE, 'stderr': sp.PIPE, 'stdin': DEVNULL}
    if os.name == 'nt':
        popen_params['creationflags'] = 134217728
    proc = sp.Popen(cmd, **popen_params)
    output, error = proc.communicate()
    infos = error.decode('utf8')
    del proc
    if print_infos:
        print(infos)
    lines = infos.splitlines()
    if 'No such file or directory' in lines[-1]:
        raise IOError('MoviePy error: the file %s could not be found!\nPlease check that you entered the correct path.' % filename)
    result = dict()
    result['duration'] = None
    if check_duration:
        try:
            keyword = 'frame=' if is_GIF else 'Duration: '
            index = -1 if is_GIF else 0
            line = [l for l in lines if keyword in l][index]
            match = re.findall('([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])', line)[0]
            result['duration'] = cvsecs(match)
        except:
            raise IOError('MoviePy error: failed to read the duration of file %s.\nHere are the file infos returned by ffmpeg:\n\n%s' % (filename, infos))
    lines_video = [l for l in lines if ' Video: ' in l and re.search('\\d+x\\d+', l)]
    result['video_found'] = lines_video != []
    if result['video_found']:
        try:
            line = lines_video[0]
            match = re.search(' [0-9]*x[0-9]*(,| )', line)
            s = list(map(int, line[match.start():match.end() - 1].split('x')))
            result['video_size'] = s
        except:
            raise IOError('MoviePy error: failed to read video dimensions in file %s.\nHere are the file infos returned by ffmpeg:\n\n%s' % (filename, infos))

        def get_tbr():
            match = re.search('( [0-9]*.| )[0-9]* tbr', line)
            s_tbr = line[match.start():match.end()].split(' ')[1]
            if 'k' in s_tbr:
                tbr = float(s_tbr.replace('k', '')) * 1000
            else:
                tbr = float(s_tbr)
            return tbr

        def get_fps():
            match = re.search('( [0-9]*.| )[0-9]* fps', line)
            fps = float(line[match.start():match.end()].split(' ')[1])
            return fps
        if fps_source == 'tbr':
            try:
                result['video_fps'] = get_tbr()
            except:
                result['video_fps'] = get_fps()
        elif fps_source == 'fps':
            try:
                result['video_fps'] = get_fps()
            except:
                result['video_fps'] = get_tbr()
        coef = 1000.0 / 1001.0
        fps = result['video_fps']
        for x in [23, 24, 25, 30, 50]:
            if fps != x and abs(fps - x * coef) < 0.01:
                result['video_fps'] = x * coef
        if check_duration:
            result['video_nframes'] = int(result['duration'] * result['video_fps']) + 1
            result['video_duration'] = result['duration']
        else:
            result['video_nframes'] = 1
            result['video_duration'] = None
        try:
            rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\\d+$', l)]
            if len(rotation_lines):
                rotation_line = rotation_lines[0]
                match = re.search('\\d+$', rotation_line)
                result['video_rotation'] = int(rotation_line[match.start():match.end()])
            else:
                result['video_rotation'] = 0
        except:
            raise IOError('MoviePy error: failed to read video rotation in file %s.\nHere are the file infos returned by ffmpeg:\n\n%s' % (filename, infos))
    lines_audio = [l for l in lines if ' Audio: ' in l]
    result['audio_found'] = lines_audio != []
    if result['audio_found']:
        line = lines_audio[0]
        try:
            match = re.search(' [0-9]* Hz', line)
            hz_string = line[match.start() + 1:match.end() - 3]
            result['audio_fps'] = int(hz_string)
        except:
            result['audio_fps'] = 'unknown'
    return result