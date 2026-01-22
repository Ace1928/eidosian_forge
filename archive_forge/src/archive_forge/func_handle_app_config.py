from django.core.management.base import AppCommand
from django.db import DEFAULT_DB_ALIAS, connections
def handle_app_config(self, app_config, **options):
    if app_config.models_module is None:
        return
    connection = connections[options['database']]
    models = app_config.get_models(include_auto_created=True)
    statements = connection.ops.sequence_reset_sql(self.style, models)
    if not statements and options['verbosity'] >= 1:
        self.stderr.write('No sequences found.')
    return '\n'.join(statements)