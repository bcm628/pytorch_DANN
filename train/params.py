from models import models

# utility params
fig_mode = None
embed_plot_epoch=10

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)
modality = None

#TODO: change batch_size
batch_size = 20
epochs = 1000
gamma = 10
theta = 1

mod_dim = -1
padding_len = 50
output_dim = -1
extractor_layers = 1
domain_layers = 1
class_layers = 1

#keep these the same as FMT
proj_dim_a = 40
proj_dim_v = 80

emo_labels = ['happy', 'sad', 'angry']

# path params
data_root = './data'

iemocap_path = 'C:/Users/bcmye/PycharmProjects/dissertation/Data/IEMOCAP_aligned'
mosei_path = 'C:/Users/bcmye/PycharmProjects/CMU-MultimodalSDK/data/MOSEI_aligned/'
# mnist_path = data_root + '/MNIST'
# mnistm_path = data_root + '/MNIST_M'
# svhn_path = data_root + '/SVHN'
# syndig_path = data_root + '/SynthDigits'

save_dir = './experiment'


