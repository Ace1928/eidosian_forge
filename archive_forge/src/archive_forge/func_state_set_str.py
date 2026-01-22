def state_set_str(set):
    return '[%s]' % ','.join(['S%d' % state.number for state in set])