from .meta import MetaMessage
def fix_end_of_track(messages, skip_checks=False):
    """Remove all end_of_track messages and add one at the end.

    This is used by merge_tracks() and MidiFile.save()."""
    accum = 0
    for msg in messages:
        if msg.type == 'end_of_track':
            accum += msg.time
        elif accum:
            delta = accum + msg.time
            yield msg.copy(skip_checks=skip_checks, time=delta)
            accum = 0
        else:
            yield msg
    yield MetaMessage('end_of_track', time=accum)