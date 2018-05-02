from torch import nn

class Net(nn.Module):
  def __init__(self, layer_sizes, activ):
    """
    In the constructor we instantiate nn modules and assign them as
    member variables.
    """
    super(Net, self).__init__()
    if activ == 'sigmoid':
        activation = nn.Sigmoid()
    elif activ == 'tanh':
        activation = nn.Tanh()
    else:
        activation = nn.ReLU()
    self.layer1 = nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), activation)
    self.layer2 = nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), activation)
    self.layer3 = nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), activation)
    self.layer4 = nn.Linear(layer_sizes[3], layer_sizes[4], bias=False) # No biases in the final linear layer


  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    output = self.layer1(x)
    output = output.sub(output.mean(0).expand(output.size())) # Removing mean from tensor
    output = self.layer2(output)
    output = output.sub(output.mean(0).expand(output.size())) # Removing mean from tensor
    output = self.layer3(output)
    output = output.sub(output.mean(0).expand(output.size())) # Removing mean from tensor
    output = self.layer4(output)
    output = output.sub(output.mean(0).expand(output.size())) # Removing mean from tensor
    return output
