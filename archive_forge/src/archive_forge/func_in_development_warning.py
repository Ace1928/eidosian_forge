import warnings
def in_development_warning(module):
    if SHOW_IN_DEVELOPMENT_WARNING:
        warnings.warn('The module %s is in development and your are advised against using it in production.' % module, category=FutureWarning)