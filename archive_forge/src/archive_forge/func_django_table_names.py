from collections import namedtuple
def django_table_names(self, only_existing=False, include_views=True):
    """
        Return a list of all table names that have associated Django models and
        are in INSTALLED_APPS.

        If only_existing is True, include only the tables in the database.
        """
    tables = set()
    for model in self.get_migratable_models():
        if not model._meta.managed:
            continue
        tables.add(model._meta.db_table)
        tables.update((f.m2m_db_table() for f in model._meta.local_many_to_many if f.remote_field.through._meta.managed))
    tables = list(tables)
    if only_existing:
        existing_tables = set(self.table_names(include_views=include_views))
        tables = [t for t in tables if self.identifier_converter(t) in existing_tables]
    return tables