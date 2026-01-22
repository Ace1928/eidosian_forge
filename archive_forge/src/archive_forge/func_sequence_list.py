from collections import namedtuple
def sequence_list(self):
    """
        Return a list of information about all DB sequences for all models in
        all apps.
        """
    sequence_list = []
    with self.connection.cursor() as cursor:
        for model in self.get_migratable_models():
            if not model._meta.managed:
                continue
            if model._meta.swapped:
                continue
            sequence_list.extend(self.get_sequences(cursor, model._meta.db_table, model._meta.local_fields))
            for f in model._meta.local_many_to_many:
                if f.remote_field.through._meta.auto_created:
                    sequence = self.get_sequences(cursor, f.m2m_db_table())
                    sequence_list.extend(sequence or [{'table': f.m2m_db_table(), 'column': None}])
    return sequence_list