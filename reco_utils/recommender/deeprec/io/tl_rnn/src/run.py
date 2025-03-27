"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

from util import *
import pickle
import pandas as pd
import random
from model import tlrnn
from dataPrep import dataprep
import pickle
import tensorflow as tf
import os
import sys
tf.compat.v1.disable_eager_execution()

print("tensorflow:")
tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )
print("end tensorflow")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))
"""
define directory for data
"""
directory = "../data/"

################################### START TRAINING DATA PROCESSING ##########################################

"""
read dataframe and get rid of NA's
"""
df = pd.read_csv(directory + "sequential_data_train.csv", verbose=True)
df = df.fillna(0)
sequence_id_count = df['sequence_ID'].nunique()
user_id_count = df['user_ID'].nunique()
# 打印结果
print(f"种类数量 - sequence_id: {sequence_id_count}")
print(f"种类数量 - user_id: {user_id_count}")

df = df.iloc[:50000,:] ############ !!!!!!! test
del df['eventData_3']
"""
create dataprep object. in this object all hyperparameters for dataprocessing and sampling are stored
please double check all parameters in config.py before running it
"""
sequence_id_count = df['sequence_ID'].nunique()
user_id_count = df['user_ID'].nunique()
# 打印结果
print(f"种类数量 - sequence_id: {sequence_id_count}")
print(f"种类数量 - user_id: {user_id_count}")
dataobj = dataprep(dataframe=df, id_name="user_ID", sequence_id = "sequence_ID")


"""
run preprocessor and bring datframe into propper list-like data structure for sampler
each list element is a sequence
"""
dataobj.preprocessor()


"""
evaluate cardinalities of input values
"""
dataobj.build_cardinality()


################################### START TEST DATA PROCESSING ##########################################

"""
read column-like dataframe and get rid of NA's
"""
df_test = pd.read_csv(directory + "sequential_data_test.csv")
df_test = df_test.fillna(0)

df_test = df_test[:50000] #!!!!!!!!!!!!!!!!! test
del df_test['eventData_3']

"""
create dataprep object for model test validation
"""
dataobj_test = dataprep(dataframe=df_test, id_name="user_ID", sequence_id = "sequence_ID")


"""
run preprocessor and bring datframe into propper list-like data structure for sampler
"""
dataobj_test.preprocessor()


"""
VERY IMPORTANT!!

Overwrite the feature encodings (rank-based) with the ones from the training set
Otherwise training and test are encoded differently and the predicitve power gets bad
"""
dataobj_test.ranks = dataobj.ranks


"""
sample triplets from holdout (test) set to test re-identification score
"""
dataobj_test.sampler(triplets_per_user=8, users_per_file="all")



################################### START TRAINING PROCESS ##########################################


"""
define cells and build tl-rnn keras model
put in dataobj, where also the pre-processed data-set
is included for "online" generator sampling
"""
my_tlrnn = tlrnn(cells=128, dataobj=dataobj)
my_tlrnn.build()


"""
alpha: seperation strength (regularizer) for triplet seperation
too much: model doesn't generalize well
too weak: model doesn't discriminate well
start with alpha=1.0 and work upwards
"""
alpha=1.0
run_name = "my_run_01"

# my_tlrnn.train_generator(run_name=run_name, alpha=alpha, test_set=dataobj_test.triplet_tensor)
my_tlrnn.load("./logs/" + run_name + "/final_weights.h5")
"""
Now, let's embedd a sequence into an embedding vector with length of cells:
"""
# take a single sequence out of the holdout data set
single_seq = dataobj_test.data_list[0][0]
# check if sequence is not longer than our model was trained for
if len(single_seq[:,0]) > my_tlrnn.sequence_len_max:
    raise Exception("Input sequence is longer than seq-len-max. Shorten the sequence and repeat.")

"""
zero-padding of the sequence to bring it to the nominal sequence length
first dim is important, but for the single sequence case just 1
"""
seq = np.zeros((1, dataobj.sequence_len_max, my_tlrnn.cov_num), dtype='int32')
seq[0, :len(single_seq[:,0]), :] = single_seq[:, :]

"""
translate the sequence into embedding vector with the trained tl_rnn model
model requires list of features
single_embedd is your embedding vector
"""
single_embedd = my_tlrnn.model_pred.predict([seq[:, :, (i):(i+1)] for i in list(range(my_tlrnn.cov_num))])
print(single_embedd)
ten_sequences = dataobj_test.data_list[0][:30]

# 检查每个序列是否超过模型训练时的最大序列长度
for single_seq in ten_sequences:
    if len(single_seq[:, 0]) > my_tlrnn.sequence_len_max:
        raise Exception("Input sequence is longer than seq-len-max. Shorten the sequence and repeat.")

# zero-padding 所有序列，将它们调整为名义上的序列长度
padded_sequences = np.zeros((len(ten_sequences), my_tlrnn.sequence_len_max, my_tlrnn.cov_num), dtype='int32')

for i, single_seq in enumerate(ten_sequences):
    padded_sequences[i, :len(single_seq[:, 0]), :] = single_seq[:, :]

padded_sequences = np.array(padded_sequences)
# 将这些序列转换成嵌入向量，使用训练好的 tl_rnn 模型
padded_sequences_constant = tf.constant(padded_sequences, dtype=tf.int32, name="item_history")

embeddings = my_tlrnn.model_pred.predict([padded_sequences_constant[:, :, (i):(i+1)] for i in range(my_tlrnn.cov_num)], steps=1)
print("over")


