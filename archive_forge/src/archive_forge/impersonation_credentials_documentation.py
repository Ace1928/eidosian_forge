from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
from oauth2client import client
from gslib.iamcredentials_api import IamcredentailsApi
Implementation of credentials that refreshes using the iamcredentials API.