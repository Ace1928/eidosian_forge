import argparse
from osc_lib.i18n import _
def validate_keys(self, keys):
    """Validate the provided keys.

        :param keys: A list of keys to validate.
        """
    valid_keys = self.required_keys | self.optional_keys
    if valid_keys:
        invalid_keys = [k for k in keys if k not in valid_keys]
        if invalid_keys:
            msg = _('Invalid keys %(invalid_keys)s specified.\nValid keys are: %(valid_keys)s')
            raise argparse.ArgumentTypeError(msg % {'invalid_keys': ', '.join(invalid_keys), 'valid_keys': ', '.join(valid_keys)})
    if self.required_keys:
        missing_keys = [k for k in self.required_keys if k not in keys]
        if missing_keys:
            msg = _('Missing required keys %(missing_keys)s.\nRequired keys are: %(required_keys)s')
            raise argparse.ArgumentTypeError(msg % {'missing_keys': ', '.join(missing_keys), 'required_keys': ', '.join(self.required_keys)})