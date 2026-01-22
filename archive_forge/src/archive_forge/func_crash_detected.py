def crash_detected(recorded_events):
    crashed = recorded_events[-1][0] != 'p.P.oe'
    return crashed