from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    if isinstance(kwargs["params"]["activations"][0], str):
        activation = _str_to_activation[kwargs["params"]["activations"][0]]
    if isinstance(kwargs["params"]["output_activation"], str):
        output_activation = _str_to_activation[kwargs["params"]["output_activation"]]
    # DONE: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    # modules = []
    # size = kwargs["params"]["layer_sizes"][0]
    # n_layers = len(kwargs["params"]["layer_sizes"])
    # modules.append(nn.Linear(input_size,size))
    # modules.append(activation)
    # for i in range(n_layers-2):
    #     modules.append(nn.Linear(size,size))
    #     modules.append(activation)
    # modules.append(nn.Linear(size,output_size))
    # modules.append(output_activation)
    # model = nn.Sequential(*modules)
    # return model
    n_layers = len(kwargs["params"]["layer_sizes"])
    size = kwargs["params"]["layer_sizes"][0]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

    # layers = []
    # in_size = input_size

    # for _ in range(len(kwargs["params"]["layer_sizes"])):
    #     layers.append(nn.Linear(in_size, kwargs["params"]["layer_sizes"][0]))
    #     layers.append(activation)
    #     in_size = kwargs["params"]["size"]

    # layers.append(nn.Linear(in_size, output_size))
    # layers.append(output_activation)
    
    # return nn.Sequential(*layers)

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
