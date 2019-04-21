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

        self.saver.restore(self.sess,'ckpt/2019-4-10-MountainCar/model.ckpt')

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

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1_policy, self.n_actions], initializer=w_initializer_policy,
                                     collections=c_names_policy)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer_policy,
                                     collections=c_names_policy)
                self.all_act = tf.matmul(l1, w2) + b2

                self.all_act_prob = tf.nn.softmax(self.all_act,
                                                  name='act_prob')  # use softmax to convert to probability

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action




