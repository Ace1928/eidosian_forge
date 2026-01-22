import sys
def save_locals(frame):
    """
    Copy values from locals_dict into the fast stack slots in the given frame.

    Note: the 'save_locals' branch had a different approach wrapping the frame (much more code, but it gives ideas
    on how to save things partially, not the 'whole' locals).
    """
    if not isinstance(frame, frame_type):
        return
    if save_locals_impl is not None:
        try:
            save_locals_impl(frame)
        except:
            pass