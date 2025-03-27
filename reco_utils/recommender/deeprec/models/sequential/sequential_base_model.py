# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import socket
import sys
from collections import defaultdict
from reco_utils.recommender.deeprec.models.base_model import BaseModel
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, \
    load_dict
__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams
        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams.values() else 1
        )
        self.hidden_size = (
            hparams.hidden_size if "hidden_size" in hparams.values() else None
        )
        self.graph = tf.Graph() if not graph else graph
        self.which_batch = 0
        with self.graph.as_default():
            self.embedding_keeps = tf.compat.v1.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        if hparams.test_dropout:
            self.embedding_keep_prob_test = 1.0 - hparams.embedding_dropout
        else:
            self.embedding_keep_prob_test = 1.0

        with tf.compat.v1.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            self.embedding_items1, self.embedding_items2, model_output = self._build_seq_graph() # ZGYQ_debug
            # model_output = self._build_seq_graph()
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return self.embedding_items1, self.embedding_items2, logit  # ZGYQ_debug
            # return logit

    def train(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel, self).train(sess, feed_dict)

    #  def batch_train(self, file_iterator, train_sess, vm, tb):
    def batch_train(self, file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.
            tb (TensorboardX): visualization manager for TensorboardX.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        model_name = self.hparams.SUMMARIES_DIR.split('/')[-3]
        for batch_data_input in file_iterator:
            if batch_data_input:
                # 获取每一个样本的长度
                item_history_tensor = list(batch_data_input.keys())[4]
                item_history = batch_data_input[item_history_tensor]
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, embeddings1, embeddings2) = step_result # ZGYQ_debug
                # (_, _, step_loss, step_data_loss) = step_result # 原
                # (_, _, step_loss, step_data_loss, summary) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _, _, _, _) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _,) = step_result
                # if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR: 改
                #     self.writer.add_summary(summary, step) 改

                # 计算一个batch下的所有用户的个性化建模差异，获取除自身外与其他用户的最大相似度总和
                # if 'need_use_pls_contact_zgyq' in model_name and step % 100 == 0:
                    # # 计算余弦相似度矩阵
                    # norms = np.linalg.norm(embeddings1, axis=1)
                    # dot_products = np.dot(embeddings1, embeddings1.T)
                    # similarity_matrix = dot_products / np.outer(norms, norms)
                    #
                    # np.fill_diagonal(similarity_matrix, -np.inf)
                    #
                    # # 计算每行的最大值
                    # k = 6  # 第 k 大，有五个重复用户
                    # partitioned_indices = np.argpartition(-similarity_matrix, k - 1, axis=1)
                    # kth_largest_values = similarity_matrix[
                    #     np.arange(similarity_matrix.shape[0]), partitioned_indices[:, k - 1]]
                    #
                    # # 计算独特性分数
                    # uniqueness_scores = 1 - kth_largest_values
                    #
                    # IU = np.mean(uniqueness_scores)
                    #
                    # stdout_backup = sys.stdout  # 加
                    # with open(f'../../../SIGIR21-SURGE/{model_name}-userSpecificity.txt', 'a') as f:
                    #     sys.stdout = f  # 重定向输出到文件
                    #     print(
                    #         "step {0:d} , userSpecificity: {1:.4f}".format(
                    #             step, IU
                    #         )
                    #     )
                    # sys.stdout = stdout_backup  # 恢复原来的输出

                    # 计算长短期兴趣的距离
                    # distance = np.linalg.norm(embeddings1 - embeddings2)




                # 输出用户交互历史记录到文件中
                # item_history_file = '../../../SIGIR21-SURGE/item_history_non_inter.txt'
                # # 将数据输出到文件
                # if not os.path.exists(item_history_file):
                #     with open(item_history_file, "w") as f:
                #         pass
                # # 将开始时间写入文件
                # with open(item_history_file, "a") as f:
                #     for row in item_history:
                #         for value in row:
                #             if value == 0:
                #                 break
                #             f.write(str(value) + "\t")
                #         f.write("\n")
                #     f.close()

                # 输出图结构到文件中
                # if step > 9000:
                #     graph_structure_file = '../../../SIGIR21-SURGE/graph_non_inter.inter'
                #     if not os.path.exists(graph_structure_file):
                #         with open(graph_structure_file, "w") as f:
                #             pass
                #     # 获取非零元素的下标
                #     with open(graph_structure_file, "a") as f:
                #         nonzero_indices = np.nonzero(embeddings1)
                #         # current_i = None  # 记录当前行的索引
                #         for i, j, k in zip(*nonzero_indices):
                #             # if current_i is None:
                #             #     current_i = i
                #             # elif current_i == i:
                #             #     f.write("\t")
                #             # elif current_i != i:
                #             #     f.write("\n")  # 打印换行符
                #             #     current_i = i
                #             f.write(f"{step-9000}\t{i}\t{j}\t{k}\n")
                #     f.close()
                #     # if step == 9500:
                #     #     sys.exit(1)
                epoch_loss += step_loss
                step += 1
                # if any(embeddings1):
                #     print("The list contains at least one True value.")
                if step % self.hparams.show_step == 0:
                    # inter_zero_indices = np.nonzero(embeddings1)
                    # if len(inter_zero_indices[0]) > 0:
                    #     print("inter_pass")
                    # else:
                    #     print("inter_break")
                    # non_inter_zero_indices = np.nonzero(embeddings2)
                    # if len(non_inter_zero_indices[0]) > 0:
                    #     print("none_inter_pass")
                    # else:
                    #     print("none_inter_break")
                    # print(embeddings1)
                    # print("inter_num_continuous: {0:.4f}".format(embeddings1))
                    # print("candidate_inter_nums_weights: {}".format(embeddings2))
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
                    stdout_backup = sys.stdout  # 加
                    # with open('../../../SIGIR21-SURGE/'+model_name+'-output.txt', 'a') as f:
                    #     sys.stdout = f  # 重定向输出到文件
                    #     # print("intra_sequence_weights: {0}".format(embeddings1))
                    #     # print("candidate_inter_nums_weights: {}".format(embeddings2))
                    #     print(
                    #         "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                    #             step, step_loss, step_data_loss
                    #         )
                    #     )
                    sys.stdout = stdout_backup  # 恢复原来的输出
                    if self.hparams.visual_type == 'epoch':
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        # tf.summary.scalar('loss',step_loss) #加
                        tb.add_scalar('loss', step_loss, step)

                if self.hparams.visual_type == 'step':
                    if step % self.hparams.visual_step == 0:
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        # tf.summary.scalar('loss',step_loss) #加
                        tb.add_scalar('loss', step_loss, step)

                        # steps validation for visualization
                        valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                        if vm != None:
                            vm.step_update_multi_lines(valid_res)  # TODO
                        for vs in valid_res:
                            # tf.summary.scalar(vs.replace('@', '_'), valid_res[vs]) #加
                            tb.add_scalar(vs.replace('@', '_'), valid_res[vs], step)

        return epoch_loss

    def fit(
            self, train_file, valid_file, valid_num_ngs, eval_metric="group_auc", vm=None, tb=None, pretrain=False
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.compat.v1.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        #  if pretrain:
        #  self.saver_emb = tf.train.Saver({'item_lookup':'item_embedding', 'user_lookup':'user_embedding'},max_to_keep=self.hparams.epochs)
        if pretrain:
            print('start saving embedding')
            if not os.path.exists(self.hparams.PRETRAIN_DIR):
                os.makedirs(self.hparams.PRETRAIN_DIR)
            #  checkpoint_emb_path = self.saver_emb.save(
            #  sess=train_sess,
            #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
            #  )
            #  graph_def = tf.get_default_graph().as_graph_def()
            var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def,
                                                                                    var_list)
            with tf.compat.v1.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print('embedding saved')

        train_sess = self.sess
        eval_info = list()
        model_name = self.hparams.SUMMARIES_DIR.split('/')[-3]
        best_metric, self.best_epoch = 0, 0
        # if 'need_use_pls_contact_zgyq' in model_name:
        #     start_time = datetime.datetime.now()
        #     stdout_backup = sys.stdout  # 加
        #     with open(f'../../../SIGIR21-SURGE/{model_name}-eachTest.txt', 'a') as f:
        #         sys.stdout = f  # 重定向输出到文件
        #         print(
        #             "Start time: {}\n".format(start_time)
        #         )
        #     sys.stdout = stdout_backup  # 恢复原来的输出
        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )

            #  epoch_loss = self.batch_train(file_iterator, train_sess, vm, tb)
            epoch_loss = self.batch_train(file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs)
            if vm != None:
                vm.step_update_line('epoch loss', epoch_loss)
            tf.summary.scalar('epoch loss', epoch_loss)  # 加
            tb.add_scalar('epoch_loss', epoch_loss, epoch)

            valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            stdout_backup = sys.stdout  # 加
            with open('../../../SIGIR21-SURGE/'+model_name+'-output.txt', 'a') as f:
                sys.stdout = f  # 重定向输出到文件
                print(
                    "eval valid at epoch {0}: {1}".format(
                        epoch,
                        ",".join(
                            [
                                "" + str(key) + ":" + str(value)
                                for key, value in valid_res.items()
                            ]
                        ),
                    )
                )
            sys.stdout = stdout_backup  # 恢复原来的输出
            # if 'need_use_pls_contact_zgyq' in model_name:
            #     test_file = '../../tests/resources/deeprec/sequential/ml_1m_final/test_data'
            #     test_num_ngs = 99
            #     res_syn = self.run_weighted_eval(test_file, num_ngs=test_num_ngs)
            #     with open(f'../../../SIGIR21-SURGE/{model_name}-eachTest.txt', 'a') as f:
            #         sys.stdout = f  # 重定向输出到文件
            #         print(
            #             "epoch %d, res_syn: %s\n" % (
            #                 epoch, str(res_syn)
            #             )
            #         )
            #     sys.stdout = stdout_backup  # 恢复原来的输出

            if self.hparams.visual_type == 'epoch':
                if vm != None:
                    vm.step_update_multi_lines(valid_res)  # TODO
                for vs in valid_res:
                    #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                    tb.add_scalar(vs.replace('@', '_'), valid_res[vs], epoch)
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))

                    if pretrain:
                        if not os.path.exists(self.hparams.PRETRAIN_DIR):
                            os.makedirs(self.hparams.PRETRAIN_DIR)
                        #  checkpoint_emb_path = self.saver_emb.save(
                        #  sess=train_sess,
                        #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
                        #  )
                        #  graph_def = tf.get_default_graph().as_graph_def()
                        var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
                        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(train_sess,
                                                                                                train_sess.graph_def,
                                                                                                var_list)
                        with tf.compat.v1.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())

                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        with open('../../../SIGIR21-SURGE/'+model_name+'-output.txt', 'a') as f:
            sys.stdout = f  # 重定向输出到文件
            print(eval_info)
            print("best epoch: {0}".format(self.best_epoch))
            print()
        sys.stdout = stdout_backup  # 恢复原来的输出
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
                filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)

    def run_weighted_eval_predict(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        user_top10_predictions = defaultdict(list)  # 记录每个用户的 top10 预测
        user_positive_scores = {}
        for batch_data_input in self.iterator.load_data_from_file(
                filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    # embeddings, step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input) # ZGYQ_debug
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess,
                                                                                                  batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                # batch >= 100
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

                # === 新增功能：统计每个用户的 Top-10 预测 ===
                if len(user_top10_predictions) < 10:
                    step_user = np.reshape(step_user, -1)  # 用户 ID
                    step_pred = np.reshape(step_pred, -1)  # 预测分数
                    step_labels = np.reshape(step_labels, -1)  # 是否为正样本
                    item_ids = np.arange(len(step_pred))  # 假设 item_id 为数据索引（可改成真实 item_id）

                    user_item_dict = defaultdict(list)
                    user_high_positive_scores = {}  # 存储每个用户的最高正样本得分

                    for u, p, l, i in zip(step_user, step_pred, step_labels, item_ids):
                        u = int(u)
                        user_item_dict[u].append((float(p), float(i), float(l)))
                        if l == 1 and p > 0.9:  # 记录用户正样本的预测概率
                            user_high_positive_scores[u] = float(p)
                            user_positive_scores[u] = float(p)

                    # 计算每个用户的 Top-10 预测
                    if len(user_positive_scores) > 0:
                        for user, items in user_item_dict.items():
                            if user in user_high_positive_scores:  # 用户有正样本大于 0.9
                                user = int(user)
                                sorted_items = sorted(items, key=lambda x: x[0], reverse=True)  # 按预测分数降序排列
                                top10_items = sorted_items[:10]  # 取前 10 个
                                user_top10_predictions[user] = top10_items  # 记录结果


        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res, user_top10_predictions, user_positive_scores

    def run_weighted_eval(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        for batch_data_input in self.iterator.load_data_from_file(
                filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    # embeddings, step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input) # ZGYQ_debug
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess,
                                                                                                  batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                # batch >= 100
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))
        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_user(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        # return sess.run([self.embedding_items1, self.iterator.users, self.pred, self.iterator.labels], feed_dict=feed_dict)  # ZGYQ_debug
        return sess.run([self.iterator.users, self.pred, self.iterator.labels],
                        feed_dict=feed_dict)

    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output], feed_dict=feed_dict)

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        with tf.compat.v1.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                    infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def infer(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).infer(sess, feed_dict)

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        # 参数 shape=[A,B] 指定了要创建的变量的形状为 A 行 B 列的张量
        with tf.compat.v1.variable_scope("embedding", initializer=self.initializer):
            self.user_lookup = tf.compat.v1.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup = tf.compat.v1.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.compat.v1.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
        print(self.hparams.FINETUNE_DIR)
        print(not self.hparams.FINETUNE_DIR)
        if self.hparams.FINETUNE_DIR:
            import pdb;
            pdb.set_trace()
            with tf.compat.v1.Session() as sess:
                # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
                #     graph_def = tf.GraphDef()
                #     graph_def.ParseFromString(f.read())
                #     sess.graph.as_default()
                #     tf.import_graph_def(graph_def, name='')
                #  tf.global_variables_initializer().run()
                output_graph_def = tf.compat.v1.GraphDef()
                with open(self.hparams.FINETUNE_DIR + "test-model.pb", "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")

                self.item_lookup = sess.graph.get_tensor_by_name('sequential/embedding/item_embedding')
                self.user_lookup = sess.graph.get_tensor_by_name('sequential/embedding/user_embedding')
            #  print(input_x.eval())

            #  output = sess.graph.get_tensor_by_name("conv/b:0")

    def cate_embedding_loop_body_outer(self, cate_sequence, size_limit, position_now, cate_A, padding_node_list):
        # cate_sequence [size, 50]
        indices = [position_now, position_now + 5, position_now + 10, position_now + 15, position_now + 20]
        position_now = position_now + 25
        selected_rows = tf.gather(cate_sequence, indices)
        cate_sequence_slice = tf.tile(selected_rows, [1, 1])
        # 记录每个节点的邻居以及对应的边的权重
        adj_dict = {}
        for seq in cate_sequence_slice:
            seq.numpy()
            for i in range(len(seq)):
                for j in range(i + 1, min(i + 2, len(seq))):
                    if seq[i] not in adj_dict:
                        adj_dict[seq[i]] = {}
                    if seq[j] not in adj_dict:
                        adj_dict[seq[j]] = {}
                    if seq[j] not in adj_dict[seq[i]]:
                        adj_dict[seq[i]][seq[j]] = 0
                        adj_dict[seq[j]][seq[i]] = 0
                    adj_dict[seq[i]][seq[j]] += 1
                    adj_dict[seq[j]][seq[i]] += 1
        node_list = sorted(adj_dict.keys())
        # 五个不同的序列为一组，每个序列长度为50，最坏的情况是五个序列满长度，没有重复cate
        temp_padded_node_list = np.pad(node_list, (0, self.hparams.max_seq_length * 5 - len(node_list)), mode='constant')
        agg_nodes_num = len(node_list)
        agg_max_cate_num = len(temp_padded_node_list)
        temp_A = np.zeros((agg_max_cate_num, agg_max_cate_num), dtype=np.int32)
        for i in range(agg_nodes_num):
            for j in range(i + 1, agg_nodes_num):
                if node_list[i] in adj_dict and node_list[j] in adj_dict[node_list[i]]:
                    temp_A[i][j] = adj_dict[node_list[i]][node_list[j]]
                    temp_A[j][i] = adj_dict[node_list[i]][node_list[j]]
        cate_A[position_now:position_now+25] = np.broadcast_to(temp_A, (25, 250, 250))
        padding_node_list[position_now:position_now+25] = np.broadcast_to(temp_padded_node_list, (25, 250))
        return cate_sequence, size_limit, position_now, cate_A, padding_node_list


    def cate_embedding_loop_cond_outer(self, cate_sequence, size_limit, position_now, cate_A):
        return position_now + 24 < size_limit
    # def cate_embedding_loop_body_inner(self, cate_A, cate_sequence_slice, ):
    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        self.user_embedding = tf.nn.embedding_lookup(
            self.user_lookup, self.iterator.users
        )
        tf.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        # (batch_size, max_history_length, embedding_dim)
        #iterator.item_history [batch_size, seq_length]
        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_history
        )
        tf.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.cates
        )
        self.cate_history_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_history
        )
        tf.summary.histogram(
            "cate_history_embedding_output", self.cate_history_embedding
        )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )
        tf.summary.histogram("target_item_embedding_output", self.target_item_embedding)

        # dropout after embedding
        self.user_embedding = self._dropout(
            self.user_embedding, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_history_embedding = self._dropout(
            self.cate_history_embedding, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.compat.v1.trainable_variables(),
            tf.compat.v1.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)
