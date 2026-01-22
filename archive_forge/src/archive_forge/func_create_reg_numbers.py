from neutron_lib.agent.common import constants
def create_reg_numbers(flow_params):
    """Replace reg_(port|net) values with defined register numbers"""
    _replace_register(flow_params, constants.REG_PORT, constants.PORT_REG_NAME)
    _replace_register(flow_params, constants.REG_NET, constants.NET_REG_NAME)
    _replace_register(flow_params, constants.REG_REMOTE_GROUP, constants.REMOTE_GROUP_REG_NAME)
    _replace_register(flow_params, constants.REG_MIN_BW, constants.MIN_BW_REG_NAME)
    _replace_register(flow_params, constants.REG_INGRESS_BW_LIMIT, constants.INGRESS_BW_LIMIT_REG_NAME)