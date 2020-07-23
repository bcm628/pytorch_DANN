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



#TODO: may have to zero pad if there are utterances shorter than the size of the kernel
#TODO: convolution layers expect (batch_size, in_channels, height of input, width of input)
#padding = int (same value used for height and weight) or tuple (first int is height and second is width)
class Extractor(nn.Module):
    """
    CNN models adapted from @bentrevett pytorch-sentiment-analysis
    """
    def __init__(self, embedding_dim, num_layers): #embedding_dim will be params.mod_dim
        super(Extractor, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=100,
                                              kernel_size=(4, embedding_dim)) for n in range(num_layers)])
#TODO: figure out required output dims for FMT
        self.fc = nn.Linear(num_layers*100, embedding_dim)

        self.dropout = nn.Dropout() #or nn.Dropout2d()

        #print(embedding_dim, num_layers)

#TODO: try training with and without Linear layer
    def forward(self, input, embedding_dim, num_layers):
        #input = [batch_size, seq_length, embedding dim]
        input = input.unsqueeze(1) #unsqueeze to add channel dimension
        #TODO: check these squeezes
        conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs]
        #x = [batch_size, n filters, seq length - filtersizes[n] +1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #input = input.expand(input.data.shape[0], 3, 28, 28)
        #return = [batch_size, 100 * 2]
        dropped = self.dropout(torch.cat(pooled, dim=1))
        #print(dropped.shape)
        final = self.fc(dropped)
        #print(final.shape)
        return final


    # def __init__(self, embedding_dim):
    #     super(Extractor, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, embedding_dim))
    #     self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, embedding_dim))
    #     self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, embedding_dim))
    #     self.dropout = nn.Dropout()
    #
    # def forward(self, input):
    #     input = input.unsqueeze(1)
    #
    #     conved1 = F.relu(self.conv1(input).squeeze(3))
    #     conved2 = F.relu(self.conv2(input).squeeze(3))
    #     conved3 = F.relu(self.conv3(input).squeeze(3))
    #
    #     pooled1 = F.max_pool1d(conved1, conved1.shape[2]).squeeze(2)
    #     pooled2 = F.max_pool1d(conved2, conved2.shape[2]).squeeze(2)
    #     pooled3 = F.max_pool1d(conved3, conved3.shape[2]).squeeze(2)
    #
    #     cat = torch.cat((pooled1, pooled2, pooled3), dim=1)
    #
    #     return self.dropout(cat)




#from @ptrblck: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#TODO: could try classifier as DNN and CNN
class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        #two hidden layers
        #TODO: fix hardcoded embed dim
        self.in_layer = nn.Linear(74, 512)
        self.hid_layer = nn.Linear(512, 256)
        self.out_layer = nn.Linear(256, 2 * 3)
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = F.relu(self.in_layer(input))
        input = self.hid_layer(self.dropout(input))
        input = F.relu(input)
        input = self.out_layer(input)

        return input



class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        #TODO: fix hardcoded embed dim
        self.in_layer = nn.Linear(74, 512)
        self.hid_layer = nn.Linear(512, 256)
        self.out_layer = nn.Linear(256, 2)
        self.dropout = nn.Dropout()

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        input = F.relu(self.in_layer(input))
        input = F.relu(self.hid_layer(input))
        #input = F.log_softmax(self.out_layer(input), 1)
        input = self.out_layer(input)
        return input

