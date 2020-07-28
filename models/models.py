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
    """
    CNN models adapted from @bentrevett pytorch-sentiment-analysis
    """
    def __init__(self, embedding_dim, num_layers = [3,4,5]): #embedding_dim will be params.mod_dim
        super(Extractor, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=100,
                                              kernel_size=(4, embedding_dim)) for n in num_layers])
#TODO: figure out required output dims for FMT
        self.fc = nn.Linear(len(num_layers)*100, embedding_dim)

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
        self.fc1 = nn.Linear(params.mod_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2 * 3)
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = self.fc2(self.dropout(input))
        input = F.relu(input)
        input = self.fc3(input)
        input = F.relu(input)
        input = self.fc4(input)

        return F.log_softmax(input, 1)

class Class_classifier_LSTM(nn.Module):

    def __init__(self):
        super(Class_classifier_LSTM, self).__init__()
        self.lstm = nn.LSTM(params.mod_dim, 256)
        self.fc = nn.Linear(256, 2 * 3)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        #(num_layers, minibatch_size, hidden_dim)
        return(torch.zeros(1, 20, 256).cuda().detach(),
               torch.zeros(1, 20, 256).cuda().detach())

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(1, 20, -1),
                                          self.hidden)
        linear_out = self.fc(lstm_out[-1].view(20, -1))

        return F.log_softmax(linear_out)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(params.mod_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout()

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = F.relu(self.fc5(input))
        input = F.relu(self.fc3(input))
        input = F.log_softmax(self.fc4(input), 1)
        #input = self.out_layer(input)
        return input

