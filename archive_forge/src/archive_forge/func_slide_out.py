from moviepy.decorators import add_mask_if_none, requires_duration
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from .CompositeVideoClip import CompositeVideoClip
@requires_duration
def slide_out(clip, duration, side):
    """ Makes the clip go away by one side of the screen.

    Only works when the clip is included in a CompositeVideoClip,
    and if the clip has the same size as the whole composition.

    Parameters
    ===========

    clip
      A video clip.

    duration
      Time taken for the clip to fully disappear.

    side
      Side of the screen where the clip goes. One of
      'top' | 'bottom' | 'left' | 'right'

    Examples
    =========

    >>> from moviepy.editor import *
    >>> clips = [... make a list of clips]
    >>> slided_clips = [CompositeVideoClip([
                            clip.fx(transfx.slide_out, duration=1, side='left')])
                        for clip in clips]
    >>> final_clip = concatenate( slided_clips, padding=-1)

    """
    w, h = clip.size
    ts = clip.duration - duration
    pos_dict = {'left': lambda t: (min(0, w * (-(t - ts) / duration)), 'center'), 'right': lambda t: (max(0, w * ((t - ts) / duration)), 'center'), 'top': lambda t: ('center', min(0, h * (-(t - ts) / duration))), 'bottom': lambda t: ('center', max(0, h * ((t - ts) / duration)))}
    return clip.set_position(pos_dict[side])