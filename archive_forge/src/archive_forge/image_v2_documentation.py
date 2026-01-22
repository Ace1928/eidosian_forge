from openstackclient.api import image_v1
Get available images

        can add limit/marker

        :param detailed:
            For v1 compatibility only, ignored as v2 is always 'detailed'
        :param public:
            Return public images if True
        :param private:
            Return private images if True
        :param community:
            Return commuity images if True
        :param shared:
            Return shared images if True

        If public, private, community and shared are all True or all False
        then all images are returned.  All arguments False is equivalent to no
        filter and all images are returned.  All arguments True is a filter
        that includes all public, private, community and shared images which
        is the same set as all images.

        http://docs.openstack.org/api/openstack-image-service/2.0/content/list-images.html
        