import logging
import sys
import traceback
import cookielib
from urlparse import parse_qs
from saml2test import CheckError
from saml2test import FatalError
from saml2test import OperationError
from saml2test.check import ERROR
from saml2test.check import ExpectedError
from saml2test.interaction import Action
from saml2test.interaction import Interaction
from saml2test.interaction import InteractionNeeded
from saml2test.opfunc import Operation
from saml2test.status import INTERACTION
from saml2test.status import STATUSCODE
def intermit(self):
    _response = self.last_response
    _last_action = None
    _same_actions = 0
    if _response.status_code >= 400:
        done = True
    else:
        done = False
    url = _response.url
    content = _response.text
    while not done:
        rdseq = []
        while _response.status_code in [302, 301, 303]:
            url = _response.headers['location']
            if url in rdseq:
                raise FatalError('Loop detected in redirects')
            else:
                rdseq.append(url)
                if len(rdseq) > 8:
                    raise FatalError(f'Too long sequence of redirects: {rdseq}')
            logger.info('HTTP %d Location: %s', _response.status_code, url)
            for_me = False
            for redirect_uri in self.my_endpoints():
                if url.startswith(redirect_uri):
                    self.client.cookiejar = self.cjar['rp']
                    for_me = True
                    try:
                        base, query = url.split('?')
                    except ValueError:
                        pass
                    else:
                        _response = parse_qs(query)
                        self.last_response = _response
                        self.last_content = _response
                        return _response
            if for_me:
                done = True
                break
            else:
                try:
                    logger.info('GET %s', url)
                    _response = self.client.send(url, 'GET')
                except Exception as err:
                    raise FatalError(f'{err}')
                content = _response.text
                logger.info('<-- CONTENT: %s', content)
                self.position = url
                self.last_content = content
                self.response = _response
                if _response.status_code >= 400:
                    done = True
                    break
        if done or url is None:
            break
        _base = url.split('?')[0]
        try:
            _spec = self.interaction.pick_interaction(_base, content)
        except InteractionNeeded:
            self.position = url
            cnt = content.replace('\n', '').replace('\t', '').replace('\r', '')
            logger.error('URL: %s', url)
            logger.error('Page Content: %s', cnt)
            raise
        except KeyError:
            self.position = url
            cnt = content.replace('\n', '').replace('\t', '').replace('\r', '')
            logger.error('URL: %s', url)
            logger.error('Page Content: %s', cnt)
            self.err_check('interaction-needed')
        if _spec == _last_action:
            _same_actions += 1
            if _same_actions >= 3:
                self.test_output.append({'status': ERROR, 'message': 'Interaction loop detection', 'url': self.position})
                raise OperationError()
        else:
            _last_action = _spec
        if len(_spec) > 2:
            logger.info('>> %s <<', _spec['page-type'])
            if _spec['page-type'] == 'login':
                self.login_page = content
        _op = Action(_spec['control'])
        try:
            _response = _op(self.client, self, url, _response, content, self.features)
            if isinstance(_response, dict):
                self.last_response = _response
                self.last_content = _response
                return _response
            content = _response.text
            self.position = url
            self.last_content = content
            self.response = _response
            if _response.status_code >= 400:
                txt = "Got status code '%s', error: %s"
                logger.error(txt, _response.status_code, content)
                self.test_output.append({'status': ERROR, 'message': txt % (_response.status_code, content), 'url': self.position})
                raise OperationError()
        except (FatalError, InteractionNeeded, OperationError):
            raise
        except Exception as err:
            self.err_check('exception', err, False)
    self.last_response = _response
    try:
        self.last_content = _response.text
    except AttributeError:
        self.last_content = None