from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def assume_role_with_saml(self, role_arn, principal_arn, saml_assertion, policy=None, duration_seconds=None):
    """
        Returns a set of temporary security credentials for users who
        have been authenticated via a SAML authentication response.
        This operation provides a mechanism for tying an enterprise
        identity store or directory to role-based AWS access without
        user-specific credentials or configuration.

        The temporary security credentials returned by this operation
        consist of an access key ID, a secret access key, and a
        security token. Applications can use these temporary security
        credentials to sign calls to AWS services. The credentials are
        valid for the duration that you specified when calling
        `AssumeRoleWithSAML`, which can be up to 3600 seconds (1 hour)
        or until the time specified in the SAML authentication
        response's `NotOnOrAfter` value, whichever is shorter.

        The maximum duration for a session is 1 hour, and the minimum
        duration is 15 minutes, even if values outside this range are
        specified.

        Optionally, you can pass an AWS IAM access policy to this
        operation. The temporary security credentials that are
        returned by the operation have the permissions that are
        associated with the access policy of the role being assumed,
        except for any permissions explicitly denied by the policy you
        pass. This gives you a way to further restrict the permissions
        for the federated user. These policies and any applicable
        resource-based policies are evaluated when calls to AWS are
        made using the temporary security credentials.

        Before your application can call `AssumeRoleWithSAML`, you
        must configure your SAML identity provider (IdP) to issue the
        claims required by AWS. Additionally, you must use AWS
        Identity and Access Management (AWS IAM) to create a SAML
        provider entity in your AWS account that represents your
        identity provider, and create an AWS IAM role that specifies
        this SAML provider in its trust policy.

        Calling `AssumeRoleWithSAML` does not require the use of AWS
        security credentials. The identity of the caller is validated
        by using keys in the metadata document that is uploaded for
        the SAML provider entity for your identity provider.

        For more information, see the following resources:


        + `Creating Temporary Security Credentials for SAML
          Federation`_ in the Using Temporary Security Credentials
          guide.
        + `SAML Providers`_ in the Using IAM guide.
        + `Configuring a Relying Party and Claims in the Using IAM
          guide. `_
        + `Creating a Role for SAML-Based Federation`_ in the Using
          IAM guide.

        :type role_arn: string
        :param role_arn: The Amazon Resource Name (ARN) of the role that the
            caller is assuming.

        :type principal_arn: string
        :param principal_arn: The Amazon Resource Name (ARN) of the SAML
            provider in AWS IAM that describes the IdP.

        :type saml_assertion: string
        :param saml_assertion: The base-64 encoded SAML authentication response
            provided by the IdP.
        For more information, see `Configuring a Relying Party and Adding
            Claims`_ in the Using IAM guide.

        :type policy: string
        :param policy:
        An AWS IAM policy in JSON format.

        The temporary security credentials that are returned by this operation
            have the permissions that are associated with the access policy of
            the role being assumed, except for any permissions explicitly
            denied by the policy you pass. These policies and any applicable
            resource-based policies are evaluated when calls to AWS are made
            using the temporary security credentials.

        The policy must be 2048 bytes or shorter, and its packed size must be
            less than 450 bytes.

        :type duration_seconds: integer
        :param duration_seconds:
        The duration, in seconds, of the role session. The value can range from
            900 seconds (15 minutes) to 3600 seconds (1 hour). By default, the
            value is set to 3600 seconds. An expiration can also be specified
            in the SAML authentication response's `NotOnOrAfter` value. The
            actual expiration time is whichever value is shorter.

        The maximum duration for a session is 1 hour, and the minimum duration
            is 15 minutes, even if values outside this range are specified.

        """
    params = {'RoleArn': role_arn, 'PrincipalArn': principal_arn, 'SAMLAssertion': saml_assertion}
    if policy is not None:
        params['Policy'] = policy
    if duration_seconds is not None:
        params['DurationSeconds'] = duration_seconds
    return self.get_object('AssumeRoleWithSAML', params, AssumedRole, verb='POST')