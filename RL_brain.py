"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
            self.tf_rs = tf.placeholder(tf.float32, [None, ], name="reward")

            # ------------------------ build policy_net ------------------------
            with tf.variable_scope('policy_net'):
                # c_names(collections_names) are the collections to store variables
                c_names_policy, n_l1_policy, w_initializer_policy, b_initializer_policy = \
                    ['policy_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

                # first layer. collections is used later when assign to target net
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, n_l1_policy], initializer=w_initializer_policy,
                                         collections=c_names_policy)
                    b1 = tf.get_variable('b1', [1, n_l1_policy], initializer=b_initializer_policy,
                                         collections=c_names_policy)
                    l1 = tf.nn.tanh(tf.matmul(self.tf_obs, w1) + b1)
                    self.variable_summaries(w1)
                    self.variable_summaries(b1)
                tf.summary.histogram('policy_net/l1/Wx_plus_b', tf.matmul(self.tf_obs, w1) + b1)
                tf.summary.histogram('policy_net/l1/output', l1)

                # second layer. collections is used later when assign to target net
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [n_l1_policy, self.n_actions], initializer=w_initializer_policy,
                                         collections=c_names_policy)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer_policy,
                                         collections=c_names_policy)
                    self.all_act = tf.matmul(l1, w2) + b2

                    self.all_act_prob = tf.nn.softmax(self.all_act,
                                                      name='act_prob')  # use softmax to convert to probability
                    self.variable_summaries(w2)
                    self.variable_summaries(b2)
                tf.summary.histogram('policy_net/l2/act_prob', self.all_act_prob)
        # # fc1
        # layer = tf.layers.dense(
        #     inputs=self.tf_obs,
        #     units=10,
        #     activation=tf.nn.tanh,  # tanh activation
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='fc1'
        # )
        # # fc2
        # all_act = tf.layers.dense(
        #     inputs=layer,
        #     units=self.n_actions,
        #     activation=None,
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='fc2'
        # )

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
        tf.summary.scalar('loss', self.loss)

        with tf.name_scope('reward'):
            self.sumReward = tf.reduce_sum(self.tf_rs)
        tf.summary.scalar('reward', self.sumReward)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram('histogram', var)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self, step):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _, loss, train_summary, value = self.sess.run([self.train_op, self.loss, self.merged, self.sumReward], feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
             self.tf_rs: self.ep_rs,
        })

        self.train_writer.add_summary(train_summary, step)  # 调用train_writer的add_summary方法将训练过程以及训练步数保存

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm, loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



