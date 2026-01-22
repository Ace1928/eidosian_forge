from moviepy.decorators import apply_to_mask
def trans_newsize(ns):
    if isinstance(ns, (int, float)):
        return [ns * w, ns * h]
    else:
        return ns