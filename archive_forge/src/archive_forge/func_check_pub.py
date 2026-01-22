from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def check_pub(self):
    """Check the publication and refresh ``self.attrs`` publication attribute.

        Returns:
            True if the publication with ``self.name`` exists, False otherwise.
        """
    pub_info = self.__get_general_pub_info()
    if not pub_info:
        return False
    self.attrs['owner'] = pub_info.get('pubowner')
    self.attrs['comment'] = pub_info.get('comment') if pub_info.get('comment') is not None else ''
    self.attrs['parameters']['publish'] = {}
    self.attrs['parameters']['publish']['insert'] = pub_info.get('pubinsert', False)
    self.attrs['parameters']['publish']['update'] = pub_info.get('pubupdate', False)
    self.attrs['parameters']['publish']['delete'] = pub_info.get('pubdelete', False)
    if pub_info.get('pubtruncate'):
        self.attrs['parameters']['publish']['truncate'] = pub_info.get('pubtruncate')
    if not pub_info.get('puballtables'):
        table_info = self.__get_tables_pub_info()
        for i, schema_and_table in enumerate(table_info):
            table_info[i] = pg_quote_identifier(schema_and_table['schema_dot_table'], 'table')
        self.attrs['tables'] = table_info
    else:
        self.attrs['alltables'] = True
    return True