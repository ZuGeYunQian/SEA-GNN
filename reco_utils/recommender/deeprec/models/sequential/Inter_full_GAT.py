# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from tensorflow.python.keras.backend import set_session
from keras.layers import GRUCell
from keras import backend as K
from copy import deepcopy
import numpy as np
from keras.layers import MultiHeadAttention, LayerNormalization, Dense
from keras.models import Model

__all__ = ["InterFullModelGAT"]
from reco_utils.recommender.deeprec.io.tl_rnn.src.TLRNNRunner import TLRNNRunner

class Convolve(tf.keras.Model):
    def __init__(self, hidden_channels):
        super(Convolve, self).__init__()

        self.Q = tf.keras.layers.Dense(units=hidden_channels, activation=tf.keras.layers.LeakyReLU())
        self.K = tf.keras.layers.Dense(units=hidden_channels, activation=tf.keras.layers.LeakyReLU())
        self.V = tf.keras.layers.Dense(units=hidden_channels, activation=tf.keras.layers.LeakyReLU())
        self.attention = tf.keras.layers.Attention()
        self.W1 = tf.keras.layers.Dense(units=hidden_channels, activation=tf.keras.layers.LeakyReLU())
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call_att(self, inputs):
        embeddings = inputs[0]  # Shape: (batch_size, num_nodes, embedding_dim)
        weights = inputs[1]  # Shape: (batch_size, num_nodes, num_nodes)
        neighbor_set = inputs[2]  # Shape: (batch_size, num_nodes, neighbor_num)

        neighbor_embeddings = tf.gather(embeddings, neighbor_set, axis=1,
                                        batch_dims=1)  # Shape: (batch_size, num_nodes, neighbor_num, embedding_dim)
        # neighbor_embeddings = self.Q1(neighbor_embeddings)
        Q = self.Q(embeddings)  # Shape: (batch_size, num_nodes, hidden_channels)
        K = self.K(neighbor_embeddings)  # Shape: (batch_size, num_nodes, neighbor_num, hidden_channels)
        V = self.V(neighbor_embeddings)  # Shape: (batch_size, num_nodes, neighbor_num, hidden_channels)

        Q = tf.tile(tf.expand_dims(Q, axis=2), [1, 1, tf.shape(neighbor_embeddings)[2], 1])
        # Attention调用方案
        attended_neighbor_hiddens = self.attention([Q, K, V])  # Shape: (batch_size, num_nodes, neighbor_num, hidden_channels)
        pooled_hiddens = tf.reduce_mean(attended_neighbor_hiddens, axis=2)  # 平均池化，也可以使用其他池化方式
        concated_hidden = tf.keras.layers.Concatenate(axis=-1)(
            [embeddings, pooled_hiddens])  # Shape: (batch_size, num_nodes, 2 * hidden_channels)
        hidden_new = self.W1(concated_hidden)  # Shape: (batch_size, num_nodes, hidden_channels)
        normalized = hidden_new / (tf.norm(hidden_new, axis=2,
                                           keepdims=True) + 1e-6)  # Shape: (batch_size, num_nodes, hidden_channels)
        normalized = self.batch_norm(normalized)  # Batch Normalization added here
        return normalized


class GRU4Rec(tf.keras.Model):
    def __init__(self, hidden_dims):
        super(GRU4Rec, self).__init__()
        assert len(hidden_dims) == 2, "hidden_dims must contain two values"

        self.gru_1 = tf.keras.layers.GRU(hidden_dims[0], return_sequences=True)
        self.gru_2 = tf.keras.layers.GRU(hidden_dims[1], return_sequences=True)
        self.dense = tf.keras.layers.Dense(hidden_dims[1])

    def call_new(self, inputs):
        # inputs: [batch_size, item_num, dims]
        output = self.gru_1(inputs)
        output = self.gru_2(output)
        output = self.dense(output)
        user_interest = tf.reduce_mean(output, axis=1)  # Aggregate over item dimension
        return user_interest


class PinSage(tf.keras.Model):

    def __init__(self, hidden_channels, edge_weights=None, graph=None):

        # hidden_channels is list containing output channels of every convolve.
        assert type(hidden_channels) is list
        if edge_weights is not None:
            self.edge_weights = edge_weights
        self.hidden_channels = hidden_channels
        super(PinSage, self).__init__()
        # create convolves for every layer.
        self.convs = list()
        for i in range(len(hidden_channels)):
            self.convs.append(Convolve(hidden_channels[i]))
        if edge_weights is None:
            for g in graph:
                # Perform PageRank on the current graph
                pr = self.pagerank(g)
                # Append the PageRank results to the list
                self.importance.append(pr)

    def call_new(self, inputs):
        # embeddings [batch,num,dims]
        # neighbor_set [batch,num,neighbor_num]
        embeddings = inputs[0]
        sample_neighbor_num = inputs[1]
        # 对 importance_matrix 进行采样，得到邻居节点索引
        # 采样操作
        top_k_values, top_k_indices = tf.nn.top_k(self.edge_weights, k=sample_neighbor_num)
        # 重塑邻居索引的形状
        node_num = tf.shape(self.edge_weights)[1]
        reshaped_indices = tf.reshape(top_k_indices, (-1, node_num, sample_neighbor_num))
        # 邻居集合
        neighbor_set = reshaped_indices
        epsilon = 1e-10
        flag = 0
        for conv in self.convs:
            embeddings = conv.call_att([embeddings + epsilon, self.edge_weights + epsilon, neighbor_set])
            embeddings = tf.keras.layers.GRU(units=self.hidden_channels[flag], return_sequences=True)(embeddings)
            flag = flag + 1
        return embeddings


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def build_new(self, inputs):
        self.kernel = self.add_weight("kernel", shape=[inputs.shape[-1], self.output_dim], dtype=tf.float32)

    def call_new(self, inputs, D, A):

        normalized_A = tf.matmul(tf.matmul(D, A), D)
        # Perform graph convolution operation
        self.build_new(inputs)
        output = tf.matmul(tf.matmul(normalized_A, inputs), self.kernel)
        if self.activation:
            output = self.activation(output)
        return output


class GCN(tf.keras.Model):
    def __init__(self, out_put_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.out_put_dims = out_put_dim
        self.gcn1 = GCNLayer(output_dim=64, activation=tf.nn.leaky_relu)
        self.gcn2 = GCNLayer(output_dim=out_put_dim)

    def compute_degree_matrix(self, A):
        # 计算每个节点的度，即每一行的和
        degree = tf.reduce_sum(A, axis=-1)
        # 计算度矩阵的倒数的平方根
        degree_inv_sqrt = 1.0 / tf.sqrt(degree + 1e-9)
        # 创建对角矩阵，每个节点的度的倒数的平方根为对角线元素
        degree_inv_sqrt_matrix = tf.linalg.diag(degree_inv_sqrt)
        return degree_inv_sqrt_matrix

    def call_new(self, inputs, A):
        D = self.compute_degree_matrix(A)
        x = self.gcn1.call_new(inputs, D, A)
        x = self.gcn2.call_new(x, D, A)
        return x


class GAT(tf.keras.Model):

    def __init__(self, hidden_channels, head, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.head = head

    def adj_to_bias(self, adj):
        return -1e9 * (1.0 - adj)

    def attn_head_new(self, seq, out_sz, bias_mat, adj_seq, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            weight_seq_fts = tf.keras.layers.Dense(out_sz, use_bias=False)
            # weight_f_1 = tf.keras.layers.Dense(adj_seq.shape[2])
            # weight_f_2 = tf.keras.layers.Dense(adj_seq.shape[2])
            weight_f_1 = tf.keras.layers.Dense(1)
            weight_f_2 = tf.keras.layers.Dense(1)
            seq_fts = weight_seq_fts(seq)
            # Compute f_1 and f_2 using the weight matrices
            # f_1 = weight_f_1(seq_fts)
            # f_2 = weight_f_2(seq_fts)

            # f_1_expanded = tf.tile(f_1, [1, 1, bias_mat.shape[2]])
            # f_2_expanded = tf.tile(f_2, [1, 1, bias_mat.shape[2]])
            # f_1_A = f_1_expanded * adj_seq
            # f_2_A = f_2_expanded * adj_seq
            # seq_fts = tf.keras.layers.Conv1D(out_sz, 1, use_bias=False)(seq)
            # # simplest self-attention possible
            f_1 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
            f_2 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs_logits = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
            coefs_adj = tf.nn.softmax(adj_seq)

            # Combine attention coefficients using a learnable parameter

            # coefs_logits_new = tf.keras.layers.Dense(tf.shape(coefs_logits)[2], activation='relu')(coefs_logits)
            # coefs_adj_new = tf.keras.layers.Dense(tf.shape(coefs_adj)[2], activation='relu')(coefs_adj)
            # coefs = coefs_logits_new * coefs_adj_new
            alpha = tf.Variable(initial_value=0.5, trainable=True)  # Learnable parameter
            coefs = alpha * coefs_logits + (1 - alpha) * coefs_adj
            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            b = tf.Variable(tf.zeros(shape=(out_sz)))
            vals = tf.matmul(coefs, seq_fts)
            ret = tf.nn.bias_add(vals, b)
            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    weight_residual = tf.keras.layers.Dense(ret.shape[-1], use_bias=False)
                    seq_resized = weight_residual(seq)
                    ret = tf.keras.layers.Add()([ret, seq_resized])
                else:
                    ret = ret + seq
            return activation(ret)  # activation

    def call_new(self, inputs, adj, adj_seq):
        adj_bias = self.adj_to_bias(adj)
        attns = inputs
        for i in range(0, len(self.hidden_channels)-1):
            attns_head = []
            for _ in range(self.head):
                attns_one = self.attn_head_new(attns, bias_mat=adj_bias, adj_seq=adj_seq,
                out_sz=self.hidden_channels[i], activation=tf.nn.elu,
                in_drop=0, coef_drop=0, residual=True)
                attns_head += [attns_one]
            attns = tf.reduce_mean(tf.stack(attns_head, 0), 0)
        return attns

    def attn_head(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            weight_seq_fts = tf.keras.layers.Dense(out_sz, use_bias=False)
            weight_f_1 = tf.keras.layers.Dense(1)
            weight_f_2 = tf.keras.layers.Dense(1)
            seq_fts = weight_seq_fts(seq)
            # Compute f_1 and f_2 using the weight matrices
            f_1 = weight_f_1(seq_fts)
            f_2 = weight_f_2(seq_fts)

            # seq_fts = tf.keras.layers.Conv1D(out_sz, 1, use_bias=False)(seq)
            # # simplest self-attention possible
            # f_1 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
            # f_2 = tf.keras.layers.Conv1D(1, 1)(seq_fts)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            b = tf.Variable(tf.zeros(shape=(out_sz)))
            vals = tf.matmul(coefs, seq_fts)
            ret = tf.nn.bias_add(vals, b)
            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    weight_residual = tf.keras.layers.Dense(ret.shape[-1], use_bias=False)
                    seq_resized = weight_residual(seq)
                    ret = tf.keras.layers.Add()([ret, seq_resized])
                else:
                    ret = ret + seq
            return activation(ret)  # activation

    def call_old(self, inputs, adj):
        adj_bias = self.adj_to_bias(adj)
        attns = inputs
        for i in range(0, len(self.hidden_channels)-1):
            attns_head = []
            for _ in range(self.head):
                attns_one = self.attn_head(attns, bias_mat=adj_bias,
                out_sz=self.hidden_channels[i], activation=tf.nn.elu,
                in_drop=0, coef_drop=0, residual=True)
                attns_head += [attns_one]
            attns = tf.reduce_mean(tf.stack(attns_head, 0), 0)
        return attns


class CustomModel(tf.keras.Model):
    def __init__(self, output_embedding_dims):
        super(CustomModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=64, return_sequences=False)
        self.dense = tf.keras.layers.Dense(output_embedding_dims, activation=None)

    def call_new(self, X):

        attention_heads = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(X, X)

        # LSTM layer to capture temporal dependencies
        lstm_output = self.lstm(attention_heads)

        # Dense layer for final prediction
        output = self.dense(lstm_output)
        return output

class InterFullModelGAT(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.relative_threshold = 0.5
        self.metric_heads = 1
        self.attention_heads = 1
        self.pool_layers = 1
        self.layer_shared = True
        self.tl_rnn = None
        if 'kwai' in socket.gethostname():
            self.pool_length = 150  # kuaishou
        else:
            self.pool_length = 30  # taobao 改 30
        super().__init__(hparams, iterator_creator, seed=None)

    def _build_seq_graph(self):
        """ SURGE Model:

            1) Interest graph: Graph construction based on metric learning
            2) Interest fusion and extraction : Graph convolution and graph pooling
            3) Prediction: Flatten pooled graph to reduced sequence
        """
        # self.tl_rnn.model_pred.predict()
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )  # X.shape = test-[500,50,40] train-[2500,50,40] [batch_size, max_seq_length, item_embedding_dim + cate_embedding_dim]
        self.mask = self.iterator.mask
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)
        self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)

        # 12.28 注释
        # 使用 tf.gather_nd 提取每个序列的最后一个物品的 embedding
        # last_item_indices = tf.stack([tf.range(tf.shape(X)[0]), self.real_sequence_length - 1], axis=1)
        # last_item_embeddings = tf.gather_nd(X, last_item_indices)
        #
        # W1 = tf.Variable(tf.random.normal([X.shape[2], X.shape[2]]))
        # # 定义权重矩阵 W2，形状为 [embedding_dim, hidden_dim]
        # W2 = tf.Variable(tf.random.normal([X.shape[2], X.shape[2]]))
        # # 定义权重向量 q，形状为 [hidden_dim, 1]
        # q = tf.Variable(tf.random.normal([X.shape[2], 1]))
        #
        # term1 = tf.matmul(X, W1)  # [batch_size, max_seq_length, hidden_dim]
        # term2 = tf.matmul(last_item_embeddings, W2)  # [batch_size, hidden_dim]
        # term2 = tf.expand_dims(term2, axis=1)  # 扩展维度以便与 term1 进行相加
        # terms_sum = term1 + term2  # [batch_size, max_seq_length, hidden_dim]
        # # 计算权重
        # activated_terms_sum = tf.nn.swish(terms_sum)
        # a = tf.matmul(activated_terms_sum, q)  # [batch_size, max_seq_length, 1]
        # # 对 embedding 序列乘以对应的权重并相加
        # weighted_embeddings = X * a  # [batch_size, max_seq_length, embedding_dim]
        # sequence_global_embedding = tf.reduce_sum(weighted_embeddings, axis=1)
        #
        # concatenated_embeddings = tf.concat([sequence_global_embedding, last_item_embeddings],
        #                                     axis=1)  # [batch_size, embedding_dim * 2]
        #
        # # 定义权重矩阵 W，形状为 [embedding_dim * 2, embedding_dim]
        # W = tf.Variable(tf.random.normal([X.shape[2] * 2, X.shape[2]]))
        #
        # # 对拼接后的 embedding 应用权重矩阵 W
        # sequence_embedding = tf.matmul(concatenated_embeddings, W)

        customModel = CustomModel(X.shape[-1])
        sequence_embedding = customModel.call_new(X)
        # attention_weights = tf.keras.layers.Dense(1, activation='softmax')(X)
        # gru_outputs = tf.reduce_sum(attention_weights * X, axis=1)



        # TLRNN训练
        # # 定义希望延长到的长度
        # target_length = 100
        # # 计算需要填充的长度
        # padding_length = target_length - tf.shape(self.iterator.item_history)[1]
        # padded_item_history = tf.pad(self.iterator.item_history, [[0, 0], [0, padding_length]])
        # padded_item_cate_history = tf.pad(self.iterator.item_cate_history, [[0, 0], [0, padding_length]])
        # # 使用 tf.concat 连接两个张量，axis=1 表示在第二维上进行连接
        # combined_tensor = tf.concat([padded_item_history[:, :, tf.newaxis], padded_item_cate_history[:, :, tf.newaxis]],
        #                             axis=2)
        # session = tf.compat.v1.keras.backend.get_session()
        # sequence_embedding = combined_tensor
        # if self.tl_rnn is None:
        #     self.tl_rnn = self._train_seq_embedding(session)
        # with self.graph.as_default():
        #     set_session(session)
        #     # 总共三个维度,[batch_size,length,dims]
        #     combined_tensor_shape = combined_tensor.shape.as_list()
        #     if combined_tensor_shape[0] is not None:
        #         sequence_embedding = self.tl_rnn.model_pred.predict([combined_tensor[:, :, (i):(i+1)] for i in range(self.tl_rnn.cov_num)], steps=1)


        # # 扩展real_sequence_length为[batch_size,1]
        # seq_len_expanded = tf.expand_dims(self.real_sequence_length, 1)
        # # 创建掩码张量，得到[batch_size,1,50]
        # bool_mask = tf.sequence_mask(seq_len_expanded, maxlen=tf.shape(X)[1])
        # # 转置，得到[batch_size,50,1]
        # bool_mask_transposed = tf.transpose(bool_mask, perm=[0, 2, 1])
        # # 复制，得到[batch_size,50,40]
        # bool_mask_final = tf.tile(bool_mask_transposed, [1, 1, tf.shape(X)[2]])

        # with tf.name_scope('core_interest_roughly'):
        #     X_interest_roughly = tf.identity(X)
        #     hidden_dims = [64, X_interest_roughly.shape[
        #         2]]  # Two layers: 64 units for the first layer, 32 units for the second layer
        #     model_gru = GRU4Rec(hidden_dims)
        #     user_interest_gru = model_gru.call_new(X_interest_roughly)

        with tf.name_scope('intra_sequence_user'):
            X_intra_sequence = tf.identity(X)

            # 1. 短期兴趣：仅使用末尾k个物品
            # 设置要提取的embedding数量
            last_k_list = [1, 3, 5, 7]  # 支持多个窗口大小
            intra_sequence_weights = tf.Variable(
                initial_value=tf.random.uniform(shape=(len(last_k_list),), minval=0.0, maxval=1.0),
                trainable=True,
                name="last_k_weights"
            )
            intra_sequence_lstm_layer = tf.keras.layers.LSTM(units=X_intra_sequence.shape[2], return_sequences=False,
                                              name="short_term_lstm")
            # 构造索引
            batch_size = tf.shape(X)[0]
            short_term_features_list = []
            for last_k in last_k_list:
                # 使用 tf.range 动态生成每个序列索引范围
                gather_indices = tf.stack([
                    tf.tile(tf.range(last_k)[tf.newaxis, :], [batch_size, 1]),  # 列的范围
                    (tf.maximum(0, self.real_sequence_length - last_k))[:, tf.newaxis] + tf.range(last_k)
                    # 确保这里的形状为 [batch_size, last_k]
                ], axis=-1)  # 索引形状: [batch_size, last_k, 2]
                # 使用 tf.gather_nd 提取最后k个embedding
                last_item_embeddings = tf.gather_nd(X, gather_indices)
                short_term_features_temp = intra_sequence_lstm_layer(last_item_embeddings)  # 输出形状为 [batch_size, 64]
                short_term_features_list.append(short_term_features_temp)

            short_term_features_stack = tf.stack(short_term_features_list,
                                                 axis=1)  # [batch_size, len(last_k_list), embedding_dim]
            intra_sequence_normalized_weights = tf.nn.softmax(intra_sequence_weights)  # 归一化权重
            short_term_features = tf.reduce_sum(short_term_features_stack * intra_sequence_normalized_weights[:, tf.newaxis],
                                           axis=1)  # 加权平均融合

            # 2. 长期兴趣：基于整个序列
            transformer_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64,
                                                                   name="long_term_transformer")
            long_term_features = transformer_layer(
                query=X_intra_sequence, value=X_intra_sequence, key=X_intra_sequence
            )  # 输出形状为 [batch_size, seq_length, embedding_dim]

            long_term_features = tf.reduce_mean(long_term_features, axis=1)

            S_intra_sequence = []
            # weighted cosine similarity
            # 权重矩阵
            # intra_sequence_weighted_tensor = tf.compat.v1.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
            # intra_sequence_weighted_tensor = intra_sequence_weighted_tensor + 1e-12
            # X_intra_fts = X_intra_sequence * tf.expand_dims(intra_sequence_weighted_tensor, 0)
            X_intra_fts = X_intra_sequence
            # 归一化，每个节点特征向量长度为1
            X_intra_fts = tf.nn.l2_normalize(X_intra_fts, dim=2, epsilon=1e-12)
            # 计算相似度矩阵
            S_intra_sequence_one = tf.matmul(X_intra_fts, tf.transpose(X_intra_fts, (0, 2, 1)))  # B*L*L
            # min-max normalization for mask
            # 目的是使得矩阵中的每个元素都在[0,1]之间
            S_min = tf.reduce_min(S_intra_sequence_one, -1, keepdims=True)
            S_max = tf.reduce_max(S_intra_sequence_one, -1, keepdims=True)
            S_one = (S_intra_sequence_one - S_min) / (S_max - S_min + 1e-10)
            S_intra_sequence += [S_one]
            S_intra_sequence = tf.reduce_mean(tf.stack(S_intra_sequence, 0), 0)
            # W_S = tf.Variable(tf.random.normal([X.shape[1], X.shape[1]]))
            # # 将权重矩阵 W 与相似度矩阵 S 相乘，得到加权后的相似度矩阵
            # S = tf.matmul(S, tf.expand_dims(W_S, 0))
            # mask invalid nodes
            # -1代表在最后一维之后增加一维，-2代表倒数第二维和最后一维中间增加一维
            S_intra_sequence = S_intra_sequence * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask,
                                                                                                       -2)
            ## Graph sparsification via seted sparseness
            # -1代表未知，即满足第一个条件的情况下，通过-1调整为能运行的状态，此处为[2500,2500]
            S_intra_sequence_flatten = tf.reshape(S_intra_sequence, [tf.shape(S_intra_sequence)[0], -1])
            sorted_S_intra_sequence_flatten = tf.sort(S_intra_sequence_flatten, direction='DESCENDING',
                                                      axis=-1)  # B*L -> B*L
            # relative ranking strategy of the entire graph
            # 记录边的数量，形状为[2500]
            intra_sequence_num_edges = tf.cast(tf.compat.v1.count_nonzero(S_intra_sequence, [1, 2]), tf.float32)  # B
            # 计算应该保留几条边
            intra_sequence_to_keep_edge = tf.cast(tf.math.ceil(intra_sequence_num_edges * 0.5), tf.int32)
            # 计算在什么得分以上的节点应该保留
            intra_sequence_threshold_score = tf.gather_nd(sorted_S_intra_sequence_flatten,
                                                          tf.expand_dims(tf.cast(intra_sequence_to_keep_edge, tf.int32),
                                                                         -1),
                                                          batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            # 把S中得分超过threshold_score的节点保留
            A_intra_sequence = tf.cast(
                tf.greater(S_intra_sequence, tf.expand_dims(tf.expand_dims(intra_sequence_threshold_score, -1), -1)),
                tf.float32)
            # 后加，保证无向图的对称性
            A_intra_sequence = tf.math.add(A_intra_sequence, tf.transpose(A_intra_sequence, perm=[0, 2, 1]))
            A_intra_sequence = tf.where(tf.math.greater(A_intra_sequence, 0), tf.ones_like(A_intra_sequence),
                                        A_intra_sequence)

            A_intra_add_all = tf.pad(A_intra_sequence, paddings=[[0, 0], [0, 3], [0, 3]], constant_values=1.0)
            X_intra_add_all = tf.concat([X_intra_sequence, tf.expand_dims(self.user_embedding, 1)], 1)
            X_intra_add_all = tf.concat([X_intra_add_all, tf.expand_dims(short_term_features, 1)], 1)
            X_intra_add_all = tf.concat([X_intra_add_all, tf.expand_dims(long_term_features, 1)], 1)


            # GCN
            intra_GCN_model = GCN(X_intra_sequence.shape[-1])
            X_intra_GCN_output = intra_GCN_model.call_new(X_intra_add_all, A_intra_add_all)
            X_intra_GCN_output = X_intra_GCN_output[:, :X_intra_sequence.shape[1], :]
            # residual
            intra_residual_matrix = tf.Variable(
                tf.random.normal(shape=(X_intra_GCN_output.shape[2], X_intra_GCN_output.shape[2])),
                trainable=True, dtype=tf.float32)
            intra_residual_bias = tf.Variable(
                tf.zeros(shape=(X_intra_GCN_output.shape[1], X_intra_GCN_output.shape[2])))
            intra_parameter_residual = tf.Variable(initial_value=1, dtype=tf.float32)
            X_intra_sequence = tf.matmul(X_intra_sequence, intra_residual_matrix)
            X_intra_sequence = X_intra_sequence + intra_residual_bias
            X_intra_sequence = tf.multiply(X_intra_sequence, intra_parameter_residual)
            intra_parameter_add_residual = tf.Variable(initial_value=0.2, dtype=tf.float32)
            intra_parameter_add_output = tf.Variable(initial_value=0.2, dtype=tf.float32)
            X_intra_sequence = tf.add(tf.multiply(X_intra_GCN_output, intra_parameter_add_output),
                                      tf.multiply(X_intra_sequence, intra_parameter_add_residual))

            # GRU [batch,50,40]
            X_intra_sequence = tf.keras.layers.GRU(units=X_intra_sequence.shape[2],
                                                   return_sequences=True)(X_intra_sequence)

            # attention [batch,50,2,40]
            X_intra_sequence_concated = tf.concat([tf.expand_dims(X_intra_sequence, axis=2),
                                                   tf.expand_dims(tf.tile(tf.expand_dims(self.user_embedding, axis=1),
                                                                          [1, X_intra_sequence.shape[1], 1]), axis=2)],
                                                  2)
            intra_Q = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1],
                                            activation=tf.keras.layers.LeakyReLU())
            intra_K = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1],
                                            activation=tf.keras.layers.LeakyReLU())
            intra_V = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1],
                                            activation=tf.keras.layers.LeakyReLU())
            intra_self_attention = tf.keras.layers.Attention()
            X_intra_sequence_attention = intra_self_attention([intra_Q(X_intra_sequence_concated),
                                                               intra_K(X_intra_sequence_concated),
                                                               intra_V(X_intra_sequence_concated)])
            # pooling [batch,50,40]
            X_intra_sequence_pooled = tf.squeeze(tf.reduce_mean(X_intra_sequence_attention, axis=2, keepdims=True),
                                                 axis=2)
            intra_hidden_layer = tf.keras.layers.Dense(units=X_intra_sequence_pooled.shape[-1], activation=tf.nn.swish)(
                X_intra_sequence_pooled)
            X_intra_sequence_output = tf.keras.layers.BatchNormalization()(intra_hidden_layer)

            _, intra_alphas = self._attention_fcn(self.target_item_embedding, X_intra_sequence_output, 'INTRA_AGRU',
                                                  False, return_alpha=True)
            _, intra_state = dynamic_rnn_dien(
                VecAttGRUCell(self.hparams.hidden_size),
                inputs=X_intra_sequence_output,
                att_scores=tf.expand_dims(intra_alphas, -1),
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="intra_gru"
            )

        with tf.name_scope('inter_sequence'):
            X_inter = tf.identity(X)
            inter_float_mask = tf.identity(self.float_mask)
            inter_sequence_embedding = tf.identity(sequence_embedding)

            # 选定最相似的x条做inter-sequence
            similarity_matrix = tf.matmul(sequence_embedding, sequence_embedding, transpose_b=True)  # [2500, 2500]

            similarity_inter_num = 1
            max_similarity_inter_num = (tf.shape(similarity_matrix)[1] - 5) // 5
            similarity_inter_num = tf.minimum(similarity_inter_num, max_similarity_inter_num)

            # 获取 top_k 的结果
            top_k_similarities = tf.math.top_k(similarity_matrix, k=similarity_inter_num * 5 + 1, sorted=True)

            # 初始化 top_indices 为第5个位置（索引 5:6）
            top_indices_num_columns = tf.shape(top_k_similarities.indices)[1]

            # 添加条件限制
            top_indices = tf.cond(
                tf.greater_equal(top_indices_num_columns, 5),  # 条件：列数 >= 5
                lambda: top_k_similarities.indices[:, 5:6],  # 如果条件满足，取第 5 列
                lambda: top_k_similarities.indices[:, 0:1]  # 如果不满足，则取自身
            )

            # 定义循环的初始条件和变量
            i = tf.constant(1)

            def similarity_inter_condition(i, top_indices):
                # 循环条件：当前索引小于 similarity_inter_num
                return tf.less(i, similarity_inter_num)

            def similarity_inter_body(i, top_indices):
                # 获取第 (i * 5 + 5) 的索引
                top_k_temp_index = top_k_similarities.indices[:, i * 5 + 5:i * 5 + 6]
                # 将新索引拼接到 top_indices
                top_indices = tf.concat([top_indices, top_k_temp_index], axis=1)
                # 更新循环变量
                return tf.add(i, 1), top_indices

            # 使用 tf.while_loop 执行循环
            _, top_indices = tf.while_loop(
                similarity_inter_condition,  # 循环条件
                similarity_inter_body,  # 循环体
                [i, top_indices],  # 初始值
                shape_invariants=[i.get_shape(), tf.TensorShape([None, None])]
            )

            tf.expand_dims(top_indices, axis=-1)

            # 根据选择的序列索引，提取对应的embedding和mask
            top_embeddings = tf.gather(X_inter, top_indices,
                                         axis=0)  # 从X_inter中提取完整的embedding [2500, 2, 50, 40]
            top_masks = tf.gather(inter_float_mask, top_indices, axis=0)  # 从mask中提取 [2500, 2, 50]
            top_inter_sequence_embeddings = tf.gather(inter_sequence_embedding, top_indices, axis=0)# 从inter_sequence_embedding中提取 [2500, 2, 50]

            top_embeddings = tf.reshape(top_embeddings, shape=[tf.shape(X_inter)[0], -1, X_inter.shape[2]])
            top_masks = tf.reshape(top_masks, shape=[tf.shape(inter_float_mask)[0], -1])
            top_inter_sequence_embeddings = tf.reshape(top_inter_sequence_embeddings, shape=[tf.shape(inter_sequence_embedding)[0], -1])

            X_inter = tf.concat([X_inter, top_embeddings], axis=1)  # [2500, 150, 40]
            inter_float_mask = tf.concat([inter_float_mask, top_masks], axis=1)  # [2500, 150]
            inter_sequence_embedding = tf.concat([inter_sequence_embedding, top_inter_sequence_embeddings], axis=1)  # [2500, 150]


            # 随机序列 inter-sequence
            # inter_num_float = tf.Variable(initial_value=5.0, trainable=True, dtype=tf.float32, name='inter_num_float')
            #
            # # Apply sigmoid to keep the value between 0 and 1
            # inter_num_continuous = tf.sigmoid(inter_num_float) * inter_num_float + 1  # Scales to [1, 6]
            # inter_num = tf.round(inter_num_continuous)
            # inter_num = tf.cast(inter_num, tf.int32)

            inter_num = 1

            # 初始化循环的状态
            i = 0

            concat_X_mask_to_shuffle = tf.concat(
                [X, tf.expand_dims(self.float_mask, axis=-1)], axis=-1
            )

            # 计算需要填充的长度
            # 因为需要同时打乱，所以需要扩充后拼接
            target_length = tf.shape(concat_X_mask_to_shuffle)[1]
            original_length = tf.shape(sequence_embedding)[1]
            pad_length = target_length - tf.shape(sequence_embedding)[1]

            # 对 sequence_embedding 进行填充
            sequence_embedding_padded = tf.pad(sequence_embedding, [[0, 0], [0, pad_length]])

            # 扩展维度并拼接
            concat_X_mask_to_shuffle = tf.concat(
                [concat_X_mask_to_shuffle, tf.expand_dims(sequence_embedding_padded, axis=-1)], axis=-1
            )

            # 定义循环体
            def loop_body(i, X_inter, inter_float_mask, inter_sequence_embedding, concat_X_mask_to_shuffle, original_length):

                # 随机打乱 concat_X_mask_to_shuffle
                concat_X_mask_to_shuffle = tf.random.shuffle(concat_X_mask_to_shuffle)  # [2500, 50, 42]

                # 拆分打乱后的矩阵
                X_shuffle = concat_X_mask_to_shuffle[:, :, :X_inter.shape[2]]  # [2500, 50, 40]
                float_mask_shuffle = tf.squeeze(concat_X_mask_to_shuffle[:, :, X_inter.shape[2]:X_inter.shape[2] + 1], axis=-1)  # [2500, 50]
                sequence_embedding_shuffle = tf.squeeze(concat_X_mask_to_shuffle[:, :, X_inter.shape[2] + 1:], axis=-1) # [2500, 50]
                sequence_embedding_trimmed = tf.slice(sequence_embedding_shuffle, [0, 0], [-1, original_length])

                # 拼接打乱后的结果到 X_inter 和 inter_float_mask
                X_inter = tf.concat([X_inter, X_shuffle], axis=1)  # [2500, <=50*(i+1), 40]
                inter_float_mask = tf.concat([inter_float_mask, float_mask_shuffle], axis=1)  # [2500, <=50*(i+1)]
                inter_sequence_embedding = tf.concat([inter_sequence_embedding, sequence_embedding_trimmed], axis=1)


                # 更新循环变量
                return i + 1, X_inter, inter_float_mask, inter_sequence_embedding, concat_X_mask_to_shuffle, original_length

            # 定义循环的终止条件
            def loop_cond(i, *_):
                return i < inter_num

            # 使用 tf.while_loop 执行循环
            _, X_inter, inter_float_mask, inter_sequence_embedding, _, _ = tf.while_loop(
                cond=loop_cond,
                body=loop_body,
                loop_vars=[i, X_inter, inter_float_mask,inter_sequence_embedding, concat_X_mask_to_shuffle, original_length],
                shape_invariants=[
                    tf.TensorShape([]),  # int类型变量
                    tf.TensorShape([None, None, 40]),  # 对应 X_inter, 允许纬度动态变化
                    tf.TensorShape([None, None]),  # 对应 inter_float_mask, 允许纬度动态变化
                    tf.TensorShape([None, None]),  # 对应 inter_sequence_embedding, 允许纬度动态变化
                    tf.TensorShape([None, None, 42]),  # 每次传入需要打乱的concat矩阵
                    tf.TensorShape([])  # int类型变量
                ]
            )

            using_in_debug1 = X
            using_in_debug2 = X

            # 12.28注释
            inter_sequence_embedding_fts = tf.expand_dims(inter_sequence_embedding, axis=1)
            inter_sequence_embedding_fts = tf.tile(inter_sequence_embedding_fts, [1, tf.shape(X_inter)[1], 1])
            inter_sequence_embedding_fts = tf.nn.l2_normalize(inter_sequence_embedding_fts, dim=2, epsilon=1e-12)
            inter_sequence_embedding_one = tf.matmul(inter_sequence_embedding_fts, tf.transpose(inter_sequence_embedding_fts, (0, 2, 1)))

            # inter_sequence_embedding_min = tf.reduce_min(inter_sequence_embedding_one, -1, keepdims=True)
            # inter_sequence_embedding_max = tf.reduce_max(inter_sequence_embedding_one, -1, keepdims=True)
            # eps = 1e-8
            # inter_sequence_embedding_one = (inter_sequence_embedding_one - inter_sequence_embedding_min) \
            #                                / (inter_sequence_embedding_max - inter_sequence_embedding_min + eps)

            S_inter = []
            # inter_weighted_tensor = tf.compat.v1.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
            # inter_weighted_tensor = inter_weighted_tensor + 1e-12
            # X_inter_fts = X_inter * tf.expand_dims(inter_weighted_tensor, 0)
            X_inter_fts = X_inter
            X_inter_fts = tf.nn.l2_normalize(X_inter_fts, dim=2, epsilon=1e-12)
            S_inter_one = tf.matmul(X_inter_fts, tf.transpose(X_inter_fts, (0, 2, 1)))  # B*L*L
            S_inter_min = tf.reduce_min(S_inter_one, -1, keepdims=True)
            S_inter_max = tf.reduce_max(S_inter_one, -1, keepdims=True)
            eps = 1e-8
            S_inter_one = (S_inter_one - S_inter_min) / (S_inter_max - S_inter_min + eps)
            S_inter += [S_inter_one]
            S_inter = tf.reduce_mean(tf.stack(S_inter, 0), 0)
            S_inter = S_inter * tf.expand_dims(inter_float_mask, -1) * tf.expand_dims(inter_float_mask, -2)

            S_inter_flatten = tf.reshape(S_inter, [tf.shape(S_inter)[0], -1])
            sorted_S_inter_flatten = tf.sort(S_inter_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
            # relative ranking strategy of the entire graph
            # 记录边的数量，形状为[2500]
            inter_num_edges = tf.cast(tf.compat.v1.count_nonzero(S_inter, [1, 2]), tf.float32)  # B
            # 计算应该保留几条边
            inter_to_keep_edge = tf.cast(tf.math.ceil(inter_num_edges * self.relative_threshold), tf.int32)
            # 计算在什么得分以上的节点应该保留
            threshold_score = tf.gather_nd(sorted_S_inter_flatten,
                                           tf.expand_dims(tf.cast(inter_to_keep_edge, tf.int32), -1),
                                           batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            # 把S中得分超过threshold_score的节点保留
            A_inter = tf.cast(tf.greater(S_inter, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
            # 后加，保证无向图的对称性
            A_inter = tf.math.add(A_inter, tf.transpose(A_inter, perm=[0, 2, 1]))
            A_inter = tf.where(tf.math.greater(A_inter, 0), tf.ones_like(A_inter), A_inter)

            inter_sequence_embedding_A = A_inter * inter_sequence_embedding_one

            #现inter_sequence
            inter_parameter_add_residual = tf.Variable(initial_value=0.2, dtype=tf.float32)
            inter_parameter_add_output = tf.Variable(initial_value=0.2, dtype=tf.float32)
            # pinsage = PinSage([256, X.shape[2] * inter_num], S_inter)  #
            # X_inter_output = pinsage.call_new([X_inter, tf.cast(A_inter_normalized.shape[2] * 0.08, tf.int32)])
            # X_inter_output = pinsage.call_new([X_inter, tf.cast((inter_num+1)*50*0.03, tf.int32)])
            # X_inter_output = pinsage.call_new([X_inter, 10])
            inter_GAT_model = GAT([512, 256, X.shape[2] * (inter_num + similarity_inter_num)], head=1)
            # X_inter_output = inter_GAT_model.call_old(X_inter, A_inter)
            X_inter_output = inter_GAT_model.call_new(X_inter, A_inter, inter_sequence_embedding_A)
            inter_residual_matrix = tf.Variable(tf.random.normal(shape=(X.shape[2], X.shape[2])),
                                                trainable=True, dtype=tf.float32)
            inter_residual_bias = tf.Variable(tf.zeros(shape=(X.shape[1], X.shape[2])))
            inter_parameter_residual = tf.Variable(initial_value=1, dtype=tf.float32)

            # inter 修改x 原残差链接
            X = tf.matmul(X, inter_residual_matrix)
            X = X + inter_residual_bias
            X = tf.multiply(X, inter_parameter_residual)

            # residual_layer = tf.keras.layers.Dense(units=X.shape[2], activation='relu', use_bias=True)
            # X = residual_layer(X)

            # 增加一层隐藏层
            hidden_layer1 = tf.keras.layers.Dense(units=256, activation=tf.nn.swish)(X_inter_output)
            hidden_layer1 = tf.keras.layers.BatchNormalization()(hidden_layer1)
            # 增加一层输出层
            hidden_layer2 = tf.keras.layers.Dense(units=128)(hidden_layer1)
            hidden_layer2 = tf.keras.layers.LeakyReLU(alpha=0.2)(hidden_layer2)
            hidden_layer2 = tf.keras.layers.BatchNormalization()(hidden_layer2)
            # hidden_layer3 = tf.keras.layers.Dense(units=128, activation='tanh')(hidden_layer2)
            # 增加一层输出层
            output_layer = tf.keras.layers.Dense(units=X.shape[2], activation='sigmoid')(hidden_layer2)
            output_layer = tf.keras.layers.BatchNormalization()(output_layer)
            X_inter_output_4 = output_layer[:, :X.shape[1], :]

            # inter 修改X
            X = tf.add(tf.multiply(X_inter_output_4[:, :X.shape[1], :], inter_parameter_add_output),
                       tf.multiply(X, inter_parameter_add_residual))

            # use cluster score as attention weights in AUGRU
            _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(self.hparams.hidden_size),
                inputs=X,
                att_scores=tf.expand_dims(alphas, -1),
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru"
            )

        with tf.name_scope('final_concat'):
            # model_output = tf.concat(  # 消融实验
            #     [final_state, self.target_item_embedding], 1)
            model_output = tf.concat(  # [batch_size,160]
                [final_state, intra_state, self.target_item_embedding], 1)
        return self.user_embedding, self.user_embedding, model_output

    def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W
            reuse (obj): Reusing variable W in query operation
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.compat.v1.variable_scope("attention_fcn" + str(name), reuse=reuse):
            query_size = query.shape.as_list()[-1]
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.compat.v1.get_variable(
                name="attention_mat" + str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])

            if query.shape.ndims != att_inputs.shape.ndims:
                queries = tf.reshape(
                    tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
                )
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = key_value * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights

    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling

        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation

        Returns:
            X (obj): Aggerated cluster embedding
            A (obj): Pooled adjacency matrix
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer

        """
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (
                    tf.ones([A.shape.as_list()[1], A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(
                A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1)  # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon()  # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0, 2, 1))  # B*L*L / B*1*L / B*L*1
            X_q = tf.matmul(A, tf.matmul(A, X))  # B*L*F

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_layer_' + str(layer) + '_' + str(i), False,
                                                 return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_' + str(layer) + '_' + str(i),
                                                 False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_shared' + '_' + str(i), reuse, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared' + '_' + str(i), reuse,
                                                 return_alpha=True)

                ## graph attentive convolution
                E = A_bool * tf.expand_dims(f_1, 1) + A_bool * tf.transpose(tf.expand_dims(f_2, 1),
                                                                            (0, 2, 1))  # B*L*1 x B*L*1 -> B*L*L
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis=-1
                )
                Xc_one = tf.matmul(E, X)  # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.compat.v1.layers.dense(Xc_one, 40, use_bias=False)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

        with tf.name_scope('interest_extraction'):
            ## cluster fitness score
            X_q = tf.matmul(A, tf.matmul(A, Xc))  # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_layer_' + str(layer) + '_' + str(i), True,
                                                 return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc,
                                                 'f2_layer_' + str(layer) + '_' + str(i), True, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_shared' + '_' + str(i), True, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared' + '_' + str(i), True,
                                                 return_alpha=True)
                cluster_score += [f_1 + f_2]
            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(
                tf.where(boolean_mask, cluster_score, mask_paddings),
                axis=-1
            )

            ## graph pooling
            num_nodes = tf.reduce_sum(self.mask, 1)  # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool,
                               tf.cast(self.pool_length + (
                                       self.real_sequence_length - self.pool_length) / self.pool_layers * (
                                               self.pool_layers - layer - 1), tf.int32),
                               num_nodes)  # B
            cluster_score = cluster_score * self.float_mask  # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1)  # B*L
                target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1)  # B*2
                target_score = tf.gather_nd(sorted_score,
                                            target_index) + K.epsilon()  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1)  # B*L
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1),
                                            batch_dims=1) + K.epsilon()  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1))  # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(self.mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1)

            ## ensure graph connectivity
            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool),
                          tf.transpose(E, (0, 2, 1)))  # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout
            graph_readout = tf.reduce_sum(Xc * tf.expand_dims(cluster_score, -1) * tf.expand_dims(self.float_mask, -1),
                                          1)

        return Xc, A, graph_readout, cluster_score

    def _train_seq_embedding(self, session):
        return TLRNNRunner.run("/home/lab408/usr/ZGYQ/SIGIR21-SURGE/tests/resources/deeprec/sequential/ml_1m_final/", session)
