from __future__ import absolute_import
import six
from six.moves import zip
from six import BytesIO
from six.moves import http_client
from six.moves.urllib.parse import urlencode, urlparse, urljoin, urlunparse, parse_qsl
import copy
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import json
import keyword
import logging
import mimetypes
import os
import re
import httplib2
import uritemplate
import google.api_core.client_options
from google.auth.transport import mtls
from google.auth.exceptions import MutualTLSChannelError
from googleapiclient import _auth
from googleapiclient import mimeparse
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidJsonError
from googleapiclient.errors import MediaUploadSizeError
from googleapiclient.errors import UnacceptableMimeTypeError
from googleapiclient.errors import UnknownApiNameOrVersion
from googleapiclient.errors import UnknownFileType
from googleapiclient.http import build_http
from googleapiclient.http import BatchHttpRequest
from googleapiclient.http import HttpMock
from googleapiclient.http import HttpMockSequence
from googleapiclient.http import HttpRequest
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaUpload
from googleapiclient.model import JsonModel
from googleapiclient.model import MediaModel
from googleapiclient.model import RawModel
from googleapiclient.schema import Schemas
from googleapiclient._helpers import _add_query_parameter
from googleapiclient._helpers import positional
def createMethod(methodName, methodDesc, rootDesc, schema):
    """Creates a method for attaching to a Resource.

  Args:
    methodName: string, name of the method to use.
    methodDesc: object, fragment of deserialized discovery document that
      describes the method.
    rootDesc: object, the entire deserialized discovery document.
    schema: object, mapping of schema names to schema descriptions.
  """
    methodName = fix_method_name(methodName)
    pathUrl, httpMethod, methodId, accept, maxSize, mediaPathUrl = _fix_up_method_description(methodDesc, rootDesc, schema)
    parameters = ResourceMethodParameters(methodDesc)

    def method(self, **kwargs):
        for name in six.iterkeys(kwargs):
            if name not in parameters.argmap:
                raise TypeError('Got an unexpected keyword argument "%s"' % name)
        keys = list(kwargs.keys())
        for name in keys:
            if kwargs[name] is None:
                del kwargs[name]
        for name in parameters.required_params:
            if name not in kwargs:
                if name not in _PAGE_TOKEN_NAMES or _findPageTokenName(_methodProperties(methodDesc, schema, 'response')):
                    raise TypeError('Missing required parameter "%s"' % name)
        for name, regex in six.iteritems(parameters.pattern_params):
            if name in kwargs:
                if isinstance(kwargs[name], six.string_types):
                    pvalues = [kwargs[name]]
                else:
                    pvalues = kwargs[name]
                for pvalue in pvalues:
                    if re.match(regex, pvalue) is None:
                        raise TypeError('Parameter "%s" value "%s" does not match the pattern "%s"' % (name, pvalue, regex))
        for name, enums in six.iteritems(parameters.enum_params):
            if name in kwargs:
                if name in parameters.repeated_params and (not isinstance(kwargs[name], six.string_types)):
                    values = kwargs[name]
                else:
                    values = [kwargs[name]]
                for value in values:
                    if value not in enums:
                        raise TypeError('Parameter "%s" value "%s" is not an allowed value in "%s"' % (name, value, str(enums)))
        actual_query_params = {}
        actual_path_params = {}
        for key, value in six.iteritems(kwargs):
            to_type = parameters.param_types.get(key, 'string')
            if key in parameters.repeated_params and type(value) == type([]):
                cast_value = [_cast(x, to_type) for x in value]
            else:
                cast_value = _cast(value, to_type)
            if key in parameters.query_params:
                actual_query_params[parameters.argmap[key]] = cast_value
            if key in parameters.path_params:
                actual_path_params[parameters.argmap[key]] = cast_value
        body_value = kwargs.get('body', None)
        media_filename = kwargs.get('media_body', None)
        media_mime_type = kwargs.get('media_mime_type', None)
        if self._developerKey:
            actual_query_params['key'] = self._developerKey
        model = self._model
        if methodName.endswith('_media'):
            model = MediaModel()
        elif 'response' not in methodDesc:
            model = RawModel()
        headers = {}
        headers, params, query, body = model.request(headers, actual_path_params, actual_query_params, body_value)
        expanded_url = uritemplate.expand(pathUrl, params)
        url = _urljoin(self._baseUrl, expanded_url + query)
        resumable = None
        multipart_boundary = ''
        if media_filename:
            if isinstance(media_filename, six.string_types):
                if media_mime_type is None:
                    logger.warning('media_mime_type argument not specified: trying to auto-detect for %s', media_filename)
                    media_mime_type, _ = mimetypes.guess_type(media_filename)
                if media_mime_type is None:
                    raise UnknownFileType(media_filename)
                if not mimeparse.best_match([media_mime_type], ','.join(accept)):
                    raise UnacceptableMimeTypeError(media_mime_type)
                media_upload = MediaFileUpload(media_filename, mimetype=media_mime_type)
            elif isinstance(media_filename, MediaUpload):
                media_upload = media_filename
            else:
                raise TypeError('media_filename must be str or MediaUpload.')
            if media_upload.size() is not None and media_upload.size() > maxSize > 0:
                raise MediaUploadSizeError('Media larger than: %s' % maxSize)
            expanded_url = uritemplate.expand(mediaPathUrl, params)
            url = _urljoin(self._baseUrl, expanded_url + query)
            url = _fix_up_media_path_base_url(url, self._baseUrl)
            if media_upload.resumable():
                url = _add_query_parameter(url, 'uploadType', 'resumable')
            if media_upload.resumable():
                resumable = media_upload
            elif body is None:
                headers['content-type'] = media_upload.mimetype()
                body = media_upload.getbytes(0, media_upload.size())
                url = _add_query_parameter(url, 'uploadType', 'media')
            else:
                msgRoot = MIMEMultipart('related')
                setattr(msgRoot, '_write_headers', lambda self: None)
                msg = MIMENonMultipart(*headers['content-type'].split('/'))
                msg.set_payload(body)
                msgRoot.attach(msg)
                msg = MIMENonMultipart(*media_upload.mimetype().split('/'))
                msg['Content-Transfer-Encoding'] = 'binary'
                payload = media_upload.getbytes(0, media_upload.size())
                msg.set_payload(payload)
                msgRoot.attach(msg)
                fp = BytesIO()
                g = _BytesGenerator(fp, mangle_from_=False)
                g.flatten(msgRoot, unixfrom=False)
                body = fp.getvalue()
                multipart_boundary = msgRoot.get_boundary()
                headers['content-type'] = 'multipart/related; boundary="%s"' % multipart_boundary
                url = _add_query_parameter(url, 'uploadType', 'multipart')
        logger.debug('URL being requested: %s %s' % (httpMethod, url))
        return self._requestBuilder(self._http, model.response, url, method=httpMethod, body=body, headers=headers, methodId=methodId, resumable=resumable)
    docs = [methodDesc.get('description', DEFAULT_METHOD_DOC), '\n\n']
    if len(parameters.argmap) > 0:
        docs.append('Args:\n')
    skip_parameters = list(rootDesc.get('parameters', {}).keys())
    skip_parameters.extend(STACK_QUERY_PARAMETERS)
    all_args = list(parameters.argmap.keys())
    args_ordered = [key2param(s) for s in methodDesc.get('parameterOrder', [])]
    if 'body' in all_args:
        args_ordered.append('body')
    for name in sorted(all_args):
        if name not in args_ordered:
            args_ordered.append(name)
    for arg in args_ordered:
        if arg in skip_parameters:
            continue
        repeated = ''
        if arg in parameters.repeated_params:
            repeated = ' (repeated)'
        required = ''
        if arg in parameters.required_params:
            required = ' (required)'
        paramdesc = methodDesc['parameters'][parameters.argmap[arg]]
        paramdoc = paramdesc.get('description', 'A parameter')
        if '$ref' in paramdesc:
            docs.append('  %s: object, %s%s%s\n    The object takes the form of:\n\n%s\n\n' % (arg, paramdoc, required, repeated, schema.prettyPrintByName(paramdesc['$ref'])))
        else:
            paramtype = paramdesc.get('type', 'string')
            docs.append('  %s: %s, %s%s%s\n' % (arg, paramtype, paramdoc, required, repeated))
        enum = paramdesc.get('enum', [])
        enumDesc = paramdesc.get('enumDescriptions', [])
        if enum and enumDesc:
            docs.append('    Allowed values\n')
            for name, desc in zip(enum, enumDesc):
                docs.append('      %s - %s\n' % (name, desc))
    if 'response' in methodDesc:
        if methodName.endswith('_media'):
            docs.append('\nReturns:\n  The media object as a string.\n\n    ')
        else:
            docs.append('\nReturns:\n  An object of the form:\n\n    ')
            docs.append(schema.prettyPrintSchema(methodDesc['response']))
    setattr(method, '__doc__', ''.join(docs))
    return (methodName, method)