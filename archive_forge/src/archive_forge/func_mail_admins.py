from django.conf import settings
from django.core.mail.message import (
from django.core.mail.utils import DNS_NAME, CachedDnsName
from django.utils.module_loading import import_string
def mail_admins(subject, message, fail_silently=False, connection=None, html_message=None):
    """Send a message to the admins, as defined by the ADMINS setting."""
    if not settings.ADMINS:
        return
    if not all((isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.ADMINS)):
        raise ValueError('The ADMINS setting must be a list of 2-tuples.')
    mail = EmailMultiAlternatives('%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message, settings.SERVER_EMAIL, [a[1] for a in settings.ADMINS], connection=connection)
    if html_message:
        mail.attach_alternative(html_message, 'text/html')
    mail.send(fail_silently=fail_silently)