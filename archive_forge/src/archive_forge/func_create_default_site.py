from django.apps import apps as global_apps
from django.conf import settings
from django.core.management.color import no_style
from django.db import DEFAULT_DB_ALIAS, connections, router
def create_default_site(app_config, verbosity=2, interactive=True, using=DEFAULT_DB_ALIAS, apps=global_apps, **kwargs):
    try:
        Site = apps.get_model('sites', 'Site')
    except LookupError:
        return
    if not router.allow_migrate_model(using, Site):
        return
    if not Site.objects.using(using).exists():
        if verbosity >= 2:
            print('Creating example.com Site object')
        Site(pk=getattr(settings, 'SITE_ID', 1), domain='example.com', name='example.com').save(using=using)
        sequence_sql = connections[using].ops.sequence_reset_sql(no_style(), [Site])
        if sequence_sql:
            if verbosity >= 2:
                print('Resetting sequence')
            with connections[using].cursor() as cursor:
                for command in sequence_sql:
                    cursor.execute(command)