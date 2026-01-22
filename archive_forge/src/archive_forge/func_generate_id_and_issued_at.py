import os
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.receipt.providers import base
from keystone.receipt import receipt_formatters as tf
def generate_id_and_issued_at(self, receipt):
    receipt_id = self.receipt_formatter.create_receipt(receipt.user_id, receipt.methods, receipt.expires_at)
    creation_datetime_obj = self.receipt_formatter.creation_time(receipt_id)
    issued_at = ks_utils.isotime(at=creation_datetime_obj, subsecond=True)
    return (receipt_id, issued_at)