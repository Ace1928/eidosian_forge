import keyword
import re
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.models.constants import LOOKUP_SEP
def handle_inspection(self, options):
    connection = connections[options['database']]
    table_name_filter = options.get('table_name_filter')
    with connection.cursor() as cursor:
        yield '# This is an auto-generated Django model module.'
        yield "# You'll have to do the following manually to clean this up:"
        yield "#   * Rearrange models' order"
        yield '#   * Make sure each model has one field with primary_key=True'
        yield '#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior'
        yield '#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table'
        yield "# Feel free to rename the models, but don't rename db_table values or field names."
        yield ('from %s import models' % self.db_module)
        known_models = []
        types = {'t'}
        if options['include_partitions']:
            types.add('p')
        if options['include_views']:
            types.add('v')
        table_info = connection.introspection.get_table_list(cursor)
        table_info = {info.name: info for info in table_info if info.type in types}
        for table_name in options['table'] or sorted((name for name in table_info)):
            if table_name_filter is not None and callable(table_name_filter):
                if not table_name_filter(table_name):
                    continue
            try:
                try:
                    relations = connection.introspection.get_relations(cursor, table_name)
                except NotImplementedError:
                    relations = {}
                try:
                    constraints = connection.introspection.get_constraints(cursor, table_name)
                except NotImplementedError:
                    constraints = {}
                primary_key_columns = connection.introspection.get_primary_key_columns(cursor, table_name)
                primary_key_column = primary_key_columns[0] if primary_key_columns else None
                unique_columns = [c['columns'][0] for c in constraints.values() if c['unique'] and len(c['columns']) == 1]
                table_description = connection.introspection.get_table_description(cursor, table_name)
            except Exception as e:
                yield ("# Unable to inspect table '%s'" % table_name)
                yield ('# The error was: %s' % e)
                continue
            model_name = self.normalize_table_name(table_name)
            yield ''
            yield ''
            yield ('class %s(models.Model):' % model_name)
            known_models.append(model_name)
            used_column_names = []
            column_to_field_name = {}
            used_relations = set()
            for row in table_description:
                comment_notes = []
                extra_params = {}
                column_name = row.name
                is_relation = column_name in relations
                att_name, params, notes = self.normalize_col_name(column_name, used_column_names, is_relation)
                extra_params.update(params)
                comment_notes.extend(notes)
                used_column_names.append(att_name)
                column_to_field_name[column_name] = att_name
                if column_name == primary_key_column:
                    extra_params['primary_key'] = True
                    if len(primary_key_columns) > 1:
                        comment_notes.append('The composite primary key (%s) found, that is not supported. The first column is selected.' % ', '.join(primary_key_columns))
                elif column_name in unique_columns:
                    extra_params['unique'] = True
                if is_relation:
                    ref_db_column, ref_db_table = relations[column_name]
                    if extra_params.pop('unique', False) or extra_params.get('primary_key'):
                        rel_type = 'OneToOneField'
                    else:
                        rel_type = 'ForeignKey'
                        ref_pk_column = connection.introspection.get_primary_key_column(cursor, ref_db_table)
                        if ref_pk_column and ref_pk_column != ref_db_column:
                            extra_params['to_field'] = ref_db_column
                    rel_to = 'self' if ref_db_table == table_name else self.normalize_table_name(ref_db_table)
                    if rel_to in known_models:
                        field_type = '%s(%s' % (rel_type, rel_to)
                    else:
                        field_type = "%s('%s'" % (rel_type, rel_to)
                    if rel_to in used_relations:
                        extra_params['related_name'] = '%s_%s_set' % (model_name.lower(), att_name)
                    used_relations.add(rel_to)
                else:
                    field_type, field_params, field_notes = self.get_field_type(connection, table_name, row)
                    extra_params.update(field_params)
                    comment_notes.extend(field_notes)
                    field_type += '('
                if att_name == 'id' and extra_params == {'primary_key': True}:
                    if field_type == 'AutoField(':
                        continue
                    elif field_type == connection.features.introspected_field_types['AutoField'] + '(':
                        comment_notes.append('AutoField?')
                if row.null_ok:
                    extra_params['blank'] = True
                    extra_params['null'] = True
                field_desc = '%s = %s%s' % (att_name, '' if '.' in field_type else 'models.', field_type)
                if field_type.startswith(('ForeignKey(', 'OneToOneField(')):
                    field_desc += ', models.DO_NOTHING'
                if connection.features.supports_comments and row.comment:
                    extra_params['db_comment'] = row.comment
                if extra_params:
                    if not field_desc.endswith('('):
                        field_desc += ', '
                    field_desc += ', '.join(('%s=%r' % (k, v) for k, v in extra_params.items()))
                field_desc += ')'
                if comment_notes:
                    field_desc += '  # ' + ' '.join(comment_notes)
                yield ('    %s' % field_desc)
            comment = None
            if (info := table_info.get(table_name)):
                is_view = info.type == 'v'
                is_partition = info.type == 'p'
                if connection.features.supports_comments:
                    comment = info.comment
            else:
                is_view = False
                is_partition = False
            yield from self.get_meta(table_name, constraints, column_to_field_name, is_view, is_partition, comment)