import torch.nn as nn

def weights_initializer(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.normal_(model.weight.data, mean=0, std=0.01)
        nn.init.constant_(model.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, mean=0, std=0.01)
        nn.init.constant_(model.bias.data, 0.0)




