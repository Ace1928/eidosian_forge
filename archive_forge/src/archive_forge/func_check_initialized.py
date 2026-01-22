import types
import weakref
import six
from apitools.base.protorpclite import util
def check_initialized(self):
    """Check class for initialization status.

        Check that all required fields are initialized

        Raises:
          ValidationError: If message is not initialized.
        """
    for name, field in self.__by_name.items():
        value = getattr(self, name)
        if value is None:
            if field.required:
                raise ValidationError('Message %s is missing required field %s' % (type(self).__name__, name))
        else:
            try:
                if isinstance(field, MessageField) and issubclass(field.message_type, Message):
                    if field.repeated:
                        for item in value:
                            item_message_value = field.value_to_message(item)
                            item_message_value.check_initialized()
                    else:
                        message_value = field.value_to_message(value)
                        message_value.check_initialized()
            except ValidationError as err:
                if not hasattr(err, 'message_name'):
                    err.message_name = type(self).__name__
                raise