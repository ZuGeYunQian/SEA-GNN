"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

from reco_utils.recommender.deeprec.io.tl_rnn.src.util import *
import pickle
import pandas as pd
import random
from reco_utils.recommender.deeprec.io.tl_rnn.src.model import tlrnn
from reco_utils.recommender.deeprec.io.tl_rnn.src.dataPrep import dataprep
import pickle
import tensorflow as tf
import os
import sys
from tensorflow.python.keras.backend import set_session

def file_exists(file_path):
    return os.path.exists(file_path)

class TLRNNRunner:

    @staticmethod
    def run(directory, sess):
        set_session(sess)
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
        # directory = "../data/"

        ################################### START TRAINING DATA PROCESSING ##########################################

        """
        read dataframe and get rid of NA's
        """
        df = pd.read_csv(directory + "sequential_data.csv", verbose=True)
        df = df.fillna(0)

        # df = df.iloc[:50000,:] ############ !!!!!!! test
        """
        create dataprep object. in this object all hyperparameters for dataprocessing and sampling are stored
        please double check all parameters in config.py before running it
        """
        dataobj = dataprep(dataframe=df, id_name="user_ID", sequence_id="sequence_ID")


        """
        run preprocessor and bring datframe into propper list-like data structure for sampler
        each list element is a sequence
        """
        dataobj.preprocessor()


        """
        evaluate cardinalities of input values
        """
        dataobj.build_cardinality()


        # ################################### START TEST DATA PROCESSING ##########################################
        #
        # """
        # read column-like dataframe and get rid of NA's
        # """
        # df_test = pd.read_csv(directory + "data/sequential_data_test.csv")
        # df_test = df_test.fillna(0)
        #
        # df_test = df_test[:50000] #!!!!!!!!!!!!!!!!! test
        # del df_test['eventData_2']
        # del df_test['eventData_3']
        #
        # """
        # create dataprep object for model test validation
        # """
        # dataobj_test = dataprep(dataframe=df_test, id_name="user_ID", sequence_id = "sequence_ID")
        #
        #
        # """
        # run preprocessor and bring datframe into propper list-like data structure for sampler
        # """
        # dataobj_test.preprocessor()
        #
        #
        # """
        # VERY IMPORTANT!!
        #
        # Overwrite the feature encodings (rank-based) with the ones from the training set
        # Otherwise training and test are encoded differently and the predicitve power gets bad
        # """
        # dataobj_test.ranks = dataobj.ranks
        #
        #
        # """
        # sample triplets from holdout (test) set to test re-identification score
        # """
        # dataobj_test.sampler(triplets_per_user=8, users_per_file="all")



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
        run_name = "ml-1m"

        if file_exists("./logs/" + run_name + "/final_weights.h5"):
            my_tlrnn.load("./logs/" + run_name + "/final_weights.h5")
            return my_tlrnn
        my_tlrnn.train_generator(run_name=run_name, alpha=alpha, test_set=None)
        # my_tlrnn.load(directory + "src/logs/" + run_name + "/final_weights.h5")
        # """
        # Now, let's embedd a sequence into an embedding vector with length of cells:
        # """
        # # take a single sequence out of the holdout data set
        # single_seq = dataobj_test.data_list[0][0]
        # # check if sequence is not longer than our model was trained for
        # if len(single_seq[:,0]) > my_tlrnn.sequence_len_max:
        #     raise Exception("Input sequence is longer than seq-len-max. Shorten the sequence and repeat.")
        #
        # """
        # zero-padding of the sequence to bring it to the nominal sequence length
        # first dim is important, but for the single sequence case just 1
        # """
        # seq = np.zeros((1, dataobj.sequence_len_max, my_tlrnn.cov_num), dtype='int32')
        # seq[0, :len(single_seq[:,0]), :] = single_seq[:, :]
        #
        # """
        # translate the sequence into embedding vector with the trained tl_rnn model
        # model requires list of features
        # single_embedd is your embedding vector
        # """
        # single_embedd = my_tlrnn.model_pred.predict([seq[:, :, (i):(i+1)] for i in list(range(my_tlrnn.cov_num))])
        # print(single_embedd)

        return my_tlrnn

