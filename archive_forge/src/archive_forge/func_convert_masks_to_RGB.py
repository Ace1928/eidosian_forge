import decorator
from moviepy.tools import cvsecs
@decorator.decorator
def convert_masks_to_RGB(f, clip, *a, **k):
    """ If the clip is a mask, convert it to RGB before running the function """
    if clip.ismask:
        clip = clip.to_RGB()
    return f(clip, *a, **k)