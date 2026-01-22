import os
import subprocess as sp
import sys
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
def ffmpeg_merge_video_audio(video, audio, output, vcodec='copy', acodec='copy', ffmpeg_output=False, logger='bar'):
    """ merges video file ``video`` and audio file ``audio`` into one
        movie file ``output``. """
    cmd = [get_setting('FFMPEG_BINARY'), '-y', '-i', audio, '-i', video, '-vcodec', vcodec, '-acodec', acodec, output]
    subprocess_call(cmd, logger=logger)