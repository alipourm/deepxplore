'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from mimicus.tools.featureedit import FeatureDescriptor
from scipy.misc import imsave

from pdf_models import *
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
# X_test, _, names = datasets.csv2numpy('./dataset/test.csv')
# X_test = X_test.astype('float32')
# num_features = X_test.shape[1]
X_train, _, names = datasets.csv2numpy('./dataset/train.csv')
X_train = X_train.astype('float32')
num_features = X_train.shape[1]
feat_names = FeatureDescriptor.get_feature_names()
incre_idx, incre_decre_idx = init_feature_constraints(feat_names)

# define input tensor as a placeholder
input_tensor = Input(shape=(num_features,))

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Model1(input_tensor=input_tensor, load_weights=True)
model2 = Model2(input_tensor=input_tensor, load_weights=True)
model3 = Model3(input_tensor=input_tensor, load_weights=True)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs

covfile = open('coverage.csv', 'w') 
infile = open('infile.csv', 'w') 

firstTime = False
for idx in xrange(len(X_train)):
    pdf = np.expand_dims(X_train[idx], axis=0)
    model1.predict(pdf)
    cov = update_coverage(pdf, model1, model_layer_dict1, THRESHOLD)
    cov_str = map(lambda x: str(x), cov.values())
    covfile.write(','.join(cov_str) + '\n')







