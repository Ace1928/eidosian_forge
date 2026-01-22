from collections import namedtuple
def get_migratable_models(self):
    from django.apps import apps
    from django.db import router
    return (model for app_config in apps.get_app_configs() for model in router.get_migratable_models(app_config, self.connection.alias) if model._meta.can_migrate(self.connection))