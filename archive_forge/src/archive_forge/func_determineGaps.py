def determineGaps(outer, inner):
    diff = outer - inner
    if INTERNAL_ALIGNMENT == 'left':
        return (0, diff)
    elif INTERNAL_ALIGNMENT == 'right':
        return (diff, 0)
    else:
        return (diff / 2, diff / 2)