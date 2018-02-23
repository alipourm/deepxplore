'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from mimicus.tools.featureedit import FeatureDescriptor
from scipy.misc import imsave

from utils import *


from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

# # read the parameter
# # argument parsing
# parser = argparse.ArgumentParser(
#     description='Main function for difference-inducing input generation in VirusTotal/Contagio dataset')
# parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
# parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
# parser.add_argument('step', help="step size of gradient descent", type=float)
# parser.add_argument('seeds', help="number of seeds of input", type=int)
# parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
# parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
# parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
#                     choices=[0, 1, 2], default=0, type=int)





# args = parser.parse_args()
THRESHOLD=0





# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets

(x_train , y_train), (_, y_test) = mnist.load_data()



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255

x_train.dump('train')

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)
# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)


# ==============================================================================================
# start gen inputs

covfile = open('coverage.csv', 'w') 
txt = ''
firstTime = False

for idx in xrange(len(x_train)):
    pdf = np.expand_dims(x_train[idx], axis=0)
    model1.predict(pdf)
    cov = update_coverage(pdf, model1, model_layer_dict1, THRESHOLD)
    cov_str = map(lambda x: str(x), cov.values())
    if idx % 100 == 0:
        print (idx)
    txt = txt+','.join(cov_str) + '\n' 
    
covfile.write(txt)








