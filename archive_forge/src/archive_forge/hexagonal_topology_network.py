class HTN(nn.Module):
    """
    Hexagonal Topology Network (HTN) class.
    This class defines a neural network with a hexagonal topology using alternating DAN and DAS modules.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the HTN module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        """
        super(HTN, self).__init__()
        self.dan1 = DAN(input_size, output_size)
        self.das1 = DAS(output_size, output_size)
        self.dan2 = DAN(output_size, output_size)
        self.das2 = DAS(output_size, output_size)
        self.dan3 = DAN(output_size, output_size)
        self.das3 = DAS(output_size, output_size)
        self.dan4 = DAN(output_size, output_size)
        self.das4 = DAS(output_size, output_size)
        self.dan5 = DAN(output_size, output_size)
        self.das5 = DAS(output_size, output_size)
        self.dan6 = DAN(output_size, output_size)
        self.das6 = DAS(output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HTN module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the hexagonal topology network.
        """
        x = x.float()  # Ensure input is float
        o1 = self.dan1(x)
        o2 = self.das1(o1)
        o3 = self.dan2(o2)
        o4 = self.das2(o3)
        o5 = self.dan3(o4)
        o6 = self.das3(o5)
        o7 = self.dan4(o6)
        o8 = self.das4(o7)
        o9 = self.dan5(o8)
        o10 = self.das5(o9)
        o11 = self.dan6(o10)
        return o11
