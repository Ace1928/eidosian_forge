@classmethod
def format_order_key(cls, key: str):
    if key.startswith('+') or key.startswith('-'):
        direction = key[0]
        key = key[1:]
    else:
        direction = '-'
    parts = key.split('.')
    if len(parts) == 1:
        if parts[0] not in ['createdAt', 'updatedAt', 'name', 'sweep']:
            return direction + 'summary_metrics.' + parts[0]
    elif parts[0] not in ['config', 'summary_metrics', 'tags']:
        return direction + '.'.join(['summary_metrics'] + parts)
    else:
        return direction + '.'.join(parts)