from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def delete_batch(self, pk_list, using):
    """
        Set up and execute delete queries for all the objects in pk_list.

        More than one physical query may be executed if there are a
        lot of values in pk_list.
        """
    num_deleted = 0
    field = self.get_meta().pk
    for offset in range(0, len(pk_list), GET_ITERATOR_CHUNK_SIZE):
        self.clear_where()
        self.add_filter(f'{field.attname}__in', pk_list[offset:offset + GET_ITERATOR_CHUNK_SIZE])
        num_deleted += self.do_query(self.get_meta().db_table, self.where, using=using)
    return num_deleted