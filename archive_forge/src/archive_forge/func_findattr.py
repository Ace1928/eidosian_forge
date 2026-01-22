def findattr(element, xpath, namespace=None):
    return element.findtext(fixxpath(xpath=xpath, namespace=namespace))