import functools
import inspect
import warnings
def generate_message(prefix, postfix=None, message=None, version=None, removal_version=None):
    """Helper to generate a common message 'style' for deprecation helpers."""
    message_components = [prefix]
    if version:
        message_components.append(" in version '%s'" % version)
    if removal_version:
        if removal_version == '?':
            message_components.append(' and will be removed in a future version')
        else:
            message_components.append(" and will be removed in version '%s'" % removal_version)
    if postfix:
        message_components.append(postfix)
    if message:
        message_components.append(': %s' % message)
    return ''.join(message_components)