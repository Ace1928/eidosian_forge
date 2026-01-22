import os
import time
import coloredlogs
def demonstrate_colored_logging():
    """Interactively demonstrate the :mod:`coloredlogs` package."""
    decorated_levels = []
    defined_levels = coloredlogs.find_defined_levels()
    normalizer = coloredlogs.NameNormalizer()
    for name, level in defined_levels.items():
        if name != 'NOTSET':
            item = (level, normalizer.normalize_name(name))
            if item not in decorated_levels:
                decorated_levels.append(item)
    ordered_levels = sorted(decorated_levels)
    coloredlogs.install(level=os.environ.get('COLOREDLOGS_LOG_LEVEL', ordered_levels[0][1]))
    for level, name in ordered_levels:
        log_method = getattr(logger, name, None)
        if log_method:
            log_method('message with level %s (%i)', name, level)
            time.sleep(DEMO_DELAY)