import os
import subprocess as sp
import proglog
from moviepy.compat import DEVNULL
from moviepy.config import get_setting
from moviepy.decorators import requires_duration
def write_frames(self, frames_array):
    try:
        try:
            self.proc.stdin.write(frames_array.tobytes())
        except NameError:
            self.proc.stdin.write(frames_array.tostring())
    except IOError as err:
        ffmpeg_error = self.proc.stderr.read()
        error = str(err) + ('\n\nMoviePy error: FFMPEG encountered the following error while writing file %s:' % self.filename + '\n\n' + str(ffmpeg_error))
        if b'Unknown encoder' in ffmpeg_error:
            error = error + "\n\nThe audio export failed because FFMPEG didn't find the specified codec for audio encoding (%s). Please install this codec or change the codec when calling to_videofile or to_audiofile. For instance for mp3:\n   >>> to_videofile('myvid.mp4', audio_codec='libmp3lame')" % self.codec
        elif b'incorrect codec parameters ?' in ffmpeg_error:
            error = error + "\n\nThe audio export failed, possibly because the codec specified for the video (%s) is not compatible with the given extension (%s). Please specify a valid 'codec' argument in to_videofile. This would be 'libmp3lame' for mp3, 'libvorbis' for ogg..." % (self.codec, self.ext)
        elif b'encoder setup failed' in ffmpeg_error:
            error = error + '\n\nThe audio export failed, possily because the bitrate you specified was two high or too low for the video codec.'
        else:
            error = error + '\n\nIn case it helps, make sure you are using a recent version of FFMPEG (the versions in the Ubuntu/Debian repos are deprecated).'
        raise IOError(error)