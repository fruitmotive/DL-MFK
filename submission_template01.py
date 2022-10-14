from torch import nn


def create_model():
    model = nn.Sequential()
    model.add_module('l1', nn.Linear(784, 256))
    model.add_module('r1', nn.ReLU())
    model.add_module('l2', nn.Linear(256, 16))
    model.add_module('r2', nn.ReLU())
    model.add_module('l3', nn.Linear(16, 10))
    return model

def count_parameters(model):
  s = 0
  for i in model.parameters():
    if len(i.shape) == 2:
      s += i.shape[0] * i.shape[1]
    else:
      s += i.shape[0]
  return s
    

