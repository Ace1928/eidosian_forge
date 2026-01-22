from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_dynamic_obj(self, dobj=None, tags=None):
    """
        Return tag object details associated with object
        Args:
            mid: Dynamic object for specified object
            tags: List or set to which the tag objects are being added, reference is returned by the method

        Returns: Tag object details associated with the given object

        """
    if tags is None:
        tags = []
    if not (isinstance(tags, list) or isinstance(tags, set)):
        self.module.fail_json(msg="The parameter 'tags' must be of type 'list' or 'set', but type %s was passed" % type(tags))
    if dobj is None:
        return tags
    temp_tags_model = self.get_tags_for_object(dobj=dobj)
    category_service = self.api_client.tagging.Category
    add_tag = tags.append if isinstance(tags, list) else tags.add
    for tag_obj in temp_tags_model:
        add_tag({'id': tag_obj.id, 'category_name': category_service.get(tag_obj.category_id).name, 'name': tag_obj.name, 'description': tag_obj.description, 'category_id': tag_obj.category_id})
    return tags