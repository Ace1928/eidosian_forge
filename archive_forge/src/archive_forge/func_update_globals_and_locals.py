import sys
def update_globals_and_locals(updated_globals, initial_globals, frame):
    assert updated_globals is not None
    f_locals = None
    removed = set(initial_globals).difference(updated_globals)
    for key, val in updated_globals.items():
        if val is not initial_globals.get(key, _SENTINEL):
            if f_locals is None:
                f_locals = frame.f_locals
            f_locals[key] = val
    if removed:
        if f_locals is None:
            f_locals = frame.f_locals
        for key in removed:
            try:
                del f_locals[key]
            except KeyError:
                pass
    if f_locals is not None:
        save_locals(frame)