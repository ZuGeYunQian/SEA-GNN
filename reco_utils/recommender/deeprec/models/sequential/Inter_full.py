# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from keras import backend as K
from copy import deepcopy
import numpy as np
from keras.layers import MultiHeadAttention, LayerNormalization, Dense

__all__ = ["InterFullModel"]

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

    def call_new(self, inputs):
        # embeddings.shape = (batch, node number, in_channels)
        embeddings = inputs[0]
        # weights.shape = (batch, node number, node number)
        weights = inputs[1]
        # neighbor_set.shape = (batch, node number, neighbor number)
        neighbor_set = inputs[2]
        neighbor_embeddings = tf.gather(embeddings, neighbor_set, axis=1, batch_dims=1)
        # neighbor_embeddings.shape = (batch, node number, neighbor number, in channels)

        # neighbor_hiddens.shape = (batch, node_number, neighbor number, hidden channels)
        neighbor_hiddens = self.Q(neighbor_embeddings)
        # indices.shape = (batch, node number, neighbor number, 2)
        node_nums = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), axis=1), (1, tf.shape(x)[2])),
                              dtype=tf.int32))(neighbor_set)

        node_nums = tf.expand_dims(node_nums, axis=0)  # 在最前面添加一个新的维度

        # 现在expanded_A的形状是 (1, m, n)
        # 可以使用broadcasting将其与B相加，使其形状变为 (batch_size, m, n)
        node_nums = node_nums + tf.zeros_like(neighbor_set)
        # indices [batch,node_number, neighbor number, 2]
        indices = tf.stack([node_nums, neighbor_set], axis=-1)

        # neighbor_weights.shape = (batch, node number, neighbor number)
        neighbor_weights = tf.gather_nd(weights, indices, batch_dims=1)
        # neighbor_weights.shape = (batch, node number, neighbor number, 1)
        neighbor_weights = tf.expand_dims(neighbor_weights, -1)

        # weighted_sum_hidden.shape = (batch, node_number, hidden channels)
        weighted_sum_hidden = tf.math.reduce_sum(
            neighbor_hiddens * neighbor_weights,
            axis=2
        ) / (tf.math.reduce_sum(neighbor_weights, axis=2) + 1e-6)

        # concated_hidden.shape = (batch, node number, in_channels + hidden channels)
        concated_hidden = tf.keras.layers.Concatenate(axis=-1)([embeddings, weighted_sum_hidden])

        # hidden_new.shape = (batch, node number, hidden_channels)
        hidden_new = self.W(concated_hidden)

        # normalized.shape = (batch, node number, hidden_channels)
        normalized = hidden_new / (tf.norm(hidden_new, axis=2, keepdims=True) + 1e-6)

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


class InterFullModel(SequentialBaseModel):

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
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )  # X.shape = test-[500,50,40] train-[2500,50,40] [batch_size, max_seq_length, item_embedding_dim + cate_embedding_dim]
        using_in_debug1 = X
        self.mask = self.iterator.mask
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)
        self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
        # 扩展real_sequence_length为[batch_size,1]
        seq_len_expanded = tf.expand_dims(self.real_sequence_length, 1)
        # 创建掩码张量，得到[batch_size,1,50]
        bool_mask = tf.sequence_mask(seq_len_expanded, maxlen=tf.shape(X)[1])
        # 转置，得到[batch_size,50,1]
        bool_mask_transposed = tf.transpose(bool_mask, perm=[0, 2, 1])
        # 复制，得到[batch_size,50,40]
        bool_mask_final = tf.tile(bool_mask_transposed, [1, 1, tf.shape(X)[2]])

        with tf.name_scope('core_interest_roughly'):
            X_interest_roughly = tf.identity(X)
            hidden_dims = [64, X_interest_roughly.shape[2]]  # Two layers: 64 units for the first layer, 32 units for the second layer
            model_gru = GRU4Rec(hidden_dims)
            user_interest_gru = model_gru.call_new(X_interest_roughly)

        with tf.name_scope('intra_sequence_user'):
            X_intra_sequence = tf.identity(X)
            S_intra_sequence = []
            # weighted cosine similarity
            # 权重矩阵
            # intra_sequence_weighted_tensor = tf.compat.v1.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
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
            S_intra_sequence = S_intra_sequence * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            ## Graph sparsification via seted sparseness
            # -1代表未知，即满足第一个条件的情况下，通过-1调整为能运行的状态，此处为[2500,2500]
            S_intra_sequence_flatten = tf.reshape(S_intra_sequence, [tf.shape(S_intra_sequence)[0], -1])
            sorted_S_intra_sequence_flatten = tf.sort(S_intra_sequence_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
            # relative ranking strategy of the entire graph
            # 记录边的数量，形状为[2500]
            intra_sequence_num_edges = tf.cast(tf.compat.v1.count_nonzero(S_intra_sequence, [1, 2]), tf.float32)  # B
            # 计算应该保留几条边
            intra_sequence_to_keep_edge = tf.cast(tf.math.ceil(intra_sequence_num_edges * 0.5), tf.int32)
            # 计算在什么得分以上的节点应该保留
            intra_sequence_threshold_score = tf.gather_nd(sorted_S_intra_sequence_flatten, tf.expand_dims(tf.cast(intra_sequence_to_keep_edge, tf.int32), -1),
                                            batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            # 把S中得分超过threshold_score的节点保留
            A_intra_sequence = tf.cast(tf.greater(S_intra_sequence, tf.expand_dims(tf.expand_dims(intra_sequence_threshold_score, -1), -1)), tf.float32)
            # 后加，保证无向图的对称性
            A_intra_sequence = tf.math.add(A_intra_sequence, tf.transpose(A_intra_sequence, perm=[0, 2, 1]))
            A_intra_sequence = tf.where(tf.math.greater(A_intra_sequence, 0), tf.ones_like(A_intra_sequence), A_intra_sequence)
            #
            A_intra_add_user = tf.pad(A_intra_sequence, paddings=[[0, 0], [0, 1], [0, 1]], constant_values=1.0)
            X_intra_add_user = tf.concat([X_intra_sequence, tf.expand_dims(self.user_embedding, 1)], 1)
            # GCN
            intra_GCN_model = GCN(X_intra_sequence.shape[-1])
            X_intra_GCN_output = intra_GCN_model.call_new(X_intra_add_user, A_intra_add_user)
            X_intra_GCN_output = tf.keras.layers.BatchNormalization()(X_intra_GCN_output)
            X_intra_GCN_output = X_intra_GCN_output[:, :X_intra_sequence.shape[1], :]
            # residual
            intra_residual_matrix = tf.Variable(tf.random.normal(shape=(X_intra_GCN_output.shape[2], X_intra_GCN_output.shape[2])),
                                                trainable=True, dtype=tf.float32)
            intra_residual_bias = tf.Variable(tf.zeros(shape=(X_intra_GCN_output.shape[1], X_intra_GCN_output.shape[2])))
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
            X_intra_sequence = tf.keras.layers.BatchNormalization()(X_intra_sequence)
            # attention [batch,50,2,40]
            X_intra_sequence_concated = tf.concat([tf.expand_dims(X_intra_sequence, axis=2),
                                                  tf.expand_dims(tf.tile(tf.expand_dims(self.user_embedding, axis=1),
                                                                            [1, X_intra_sequence.shape[1], 1]),axis=2)], 2)
            intra_Q = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1], activation=tf.keras.layers.LeakyReLU())
            intra_K = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1], activation=tf.keras.layers.LeakyReLU())
            intra_V = tf.keras.layers.Dense(units=X_intra_sequence_concated.shape[-1], activation=tf.keras.layers.LeakyReLU())
            intra_self_attention = tf.keras.layers.Attention()
            X_intra_sequence_attention = intra_self_attention([intra_Q(X_intra_sequence_concated),
                                                              intra_K(X_intra_sequence_concated),
                                                              intra_V(X_intra_sequence_concated)])
            X_intra_sequence_attention = tf.keras.layers.BatchNormalization()(X_intra_sequence_attention)
            # pooling [batch,50,40]
            X_intra_sequence_pooled = tf.squeeze(tf.reduce_mean(X_intra_sequence_attention, axis=2, keepdims=True), axis=2)
            intra_hidden_layer = tf.keras.layers.Dense(units=X_intra_sequence_pooled.shape[-1], activation=tf.nn.swish)(X_intra_sequence_pooled)
            X_intra_sequence_output = tf.keras.layers.Dropout(rate=0.3)(intra_hidden_layer)  # 设置适当的dropout率

            _, intra_alphas = self._attention_fcn(self.target_item_embedding, X_intra_sequence_output, 'INTRA_AGRU', False, return_alpha=True)
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
            # 与5个sequence进行inter-sequence
            inter_num = 2
            for i in range(inter_num):
                X_shuffle = tf.random.shuffle(X, seed=i)
                X_inter = tf.concat([X_inter, X_shuffle], axis=1)
                float_mask_shuffle = tf.random.shuffle(self.float_mask, seed=i)
                inter_float_mask = tf.concat([inter_float_mask, float_mask_shuffle], axis=1)

            S_inter = []
            # inter_weighted_tensor = tf.compat.v1.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
            # X_inter_fts = X_inter * tf.expand_dims(inter_weighted_tensor, 0)
            X_inter_fts = X_inter
            X_inter_fts = tf.nn.l2_normalize(X_inter_fts, dim=2)
            S_inter_one = tf.matmul(X_inter_fts, tf.transpose(X_inter_fts, (0, 2, 1)))  # B*L*L
            S_inter_min = tf.reduce_min(S_inter_one, -1, keepdims=True)
            S_inter_max = tf.reduce_max(S_inter_one, -1, keepdims=True)
            eps = 1e-8
            S_inter_one = (S_inter_one - S_inter_min) / (S_inter_max - S_inter_min + eps)
            S_inter += [S_inter_one]
            S_inter = tf.reduce_mean(tf.stack(S_inter, 0), 0)
            S_inter = S_inter * tf.expand_dims(inter_float_mask, -1) * tf.expand_dims(inter_float_mask, -2)
            using_in_debug2 = S_inter
            #现inter_sequence
            inter_parameter_add_residual = tf.Variable(initial_value=0.2, dtype=tf.float32)
            inter_parameter_add_output = tf.Variable(initial_value=0.2, dtype=tf.float32)

            pinsage = PinSage([256, X.shape[2] * inter_num], S_inter)  #
            # X_inter_output = pinsage.call_new([X_inter, tf.cast(A_inter_normalized.shape[2] * 0.08, tf.int32)])
            # X_inter_output = pinsage.call_new([X_inter, tf.cast((inter_num+1)*50*0.03, tf.int32)])
            X_inter_output = pinsage.call_new([X_inter, 10])

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
            model_output = tf.concat(  # [batch_size,160]
                [final_state, intra_state, self.target_item_embedding], 1)
        return using_in_debug1, using_in_debug1, model_output

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
