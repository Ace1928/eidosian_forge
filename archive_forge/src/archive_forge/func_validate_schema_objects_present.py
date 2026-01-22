from __future__ import absolute_import, division, print_function
from collections import namedtuple
def validate_schema_objects_present(self, required_schema_objects):
    """
        Validate that attributes are set to a value that is not equal None.
        :param required_schema_objects: List of schema objects to verify. -> List
        :return: None
        """
    for schema_object in required_schema_objects:
        if schema_object not in self.schema_objects.keys():
            msg = "Required attribute '{0}' is not specified on schema instance with name {1}".format(schema_object, self.schema_name)
            self.mso.fail_json(msg=msg)