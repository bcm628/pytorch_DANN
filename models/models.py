import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from train import params

#TODO: figure out why input to classifiers is not shape of output of feature extractor

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self. conv1 = nn.Conv2d(in_channels=1,
                                out_channels=10,
                                kernel_size=3,
                                padding=(1,1))
        self.bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=3,
                               padding=(1,1))
        self.pool = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout()

    def forward(self, input):
        input = input.unsqueeze(1)#20, 1, 20, 74 (batch, channel, seq, mod_dim)
        input = self.conv1(input) # 20, 10, 20, 74
        input = F.relu(self.bn(input)) #
        input = self.pool(input) #20, 10, 10, 37
        input = F.relu(self.conv2(input))
        input = self.pool(input) #20, 20,5,18
        input = self.drop(input)
        input = input.view(-1, 12*18*20)
        #input.unsqueeze(2)
        return input

class Extractor_1(nn.Module):
    """
    CNN models adapted from @bentrevett pytorch-sentiment-analysis
    """
    def __init__(self, embedding_dim, num_layers = [3,4,5]): #embedding_dim will be params.mod_dim
        super(Extractor, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=100,
                                              kernel_size=(4, embedding_dim)) for n in num_layers])
        self.dropout = nn.Dropout() #or nn.Dropout2d()


    def forward(self, input, embedding_dim, num_layers = [3,4,5]):
        #input = [batch_size, seq_length, embedding dim]
        input = input.unsqueeze(1) #unsqueeze to add channel dimension
        #TODO: check these squeezes
        conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        dropped = self.dropout(torch.cat(pooled, dim=1))
        final = self.fc(dropped)
        #print(final.shape)
        return final


#from @ptrblck: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        self.fc1 = nn.Linear(12*18*20, 256)
        self.fc2 = nn.Linear(256, 512)
        #self.fc5 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 3)
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = self.fc2(self.dropout(input))
        input = F.relu(input)
        input = self.fc3(input)
        input = F.relu(input)
        input = self.fc4(input)

        return input
        #return F.log_softmax(input, 1)


class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(12*18*20, 256)
        self.fc2 = nn.Linear(256, 256)
        #self.fc5 = nn.Linear(512, 512)
        #self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout()

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        input = F.relu(self.fc1(self.dropout(input)))
        input = F.relu(self.fc2(input))
        #input = F.relu(self.fc5(input))
        #input = F.relu(self.fc3(input))
        input = self.fc4(input)
        #input = F.log_softmax(self.fc4(input), 1)
        #input = self.out_layer(input)
        return input

