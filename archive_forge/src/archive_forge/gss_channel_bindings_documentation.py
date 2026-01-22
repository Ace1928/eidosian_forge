import struct

        Used to send the out of band channel info as part of the authentication
        process. This is used as a way of verifying the target is who it says
        it is as this information is provided by the higher layer. In most
        cases, the CBT is just the hash of the server's TLS certificate to the
        application_data field.

        This bytes string of the packed structure is then MD5 hashed and
        included in the NTv2 response.
        