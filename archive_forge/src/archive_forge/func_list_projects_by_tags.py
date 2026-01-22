from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def list_projects_by_tags(self, filters):
    filtered_ids = []
    with sql.session_for_read() as session:
        query = session.query(sql_model.ProjectTag)
        if 'tags' in filters.keys():
            filtered_ids += self._filter_ids_by_tags(query, filters['tags'].split(','))
        if 'tags-any' in filters.keys():
            any_tags = filters['tags-any'].split(',')
            subq = query.filter(sql_model.ProjectTag.name.in_(any_tags))
            any_tags = [ptag['project_id'] for ptag in subq]
            if 'tags' in filters.keys():
                any_tags = set(any_tags) & set(filtered_ids)
            filtered_ids = any_tags
        if 'not-tags' in filters.keys():
            blacklist_ids = self._filter_ids_by_tags(query, filters['not-tags'].split(','))
            filtered_ids = self._filter_not_tags(session, filtered_ids, blacklist_ids)
        if 'not-tags-any' in filters.keys():
            any_tags = filters['not-tags-any'].split(',')
            subq = query.filter(sql_model.ProjectTag.name.in_(any_tags))
            blacklist_ids = [ptag['project_id'] for ptag in subq]
            if 'not-tags' in filters.keys():
                filtered_ids += blacklist_ids
            else:
                filtered_ids = self._filter_not_tags(session, filtered_ids, blacklist_ids)
        if not filtered_ids:
            return []
        query = session.query(sql_model.Project)
        query = query.filter(sql_model.Project.id.in_(filtered_ids))
        return [project_ref.to_dict() for project_ref in query.all() if not self._is_hidden_ref(project_ref)]