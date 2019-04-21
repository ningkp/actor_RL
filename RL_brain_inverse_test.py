"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
#MountainCar  n_state = 1, state_net_input = 2
#CartPole     n_state = 6, state_net_input = 8

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradientInverse:
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
        self.lr_S = 0.005
        self.n_state = 1
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_as_star, self.ep_rs = [], [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.saver.restore(self.sess, 'ckpt/2019-4-19-MountainCar-inverseD/model.ckpt')
        #得到state-net的参数
        state_params = self.sess.run(tf.get_collection('state_net_params'))
        #重新初始化网络
        self.sess.run(tf.global_variables_initializer())
        # print(state_params)
        # print(self.sess.run(tf.get_collection('state_net_params')))

        #将state-net参数赋值
        self.replace = [tf.assign(t, e) for t, e in zip(tf.get_collection('state_net_params'), state_params)]
        self.sess.run(self.replace)
        # print(self.sess.run(tf.get_collection('state_net_params')))
        # print(self.sess.run(tf.get_collection('policy_net2_params')))
        # print(ca)


    def _build_net(self):
        with tf.name_scope('inputs2'):
            self.tf_obs2 = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts2 = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_acts_star2 = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_star")
            self.tf_vt2 = tf.placeholder(tf.float32, [None, ], name="actions_value")
            self.tf_rs2 = tf.placeholder(tf.float32, [None, ], name="reward")

        # ------------------------ build state_net ------------------------
        with tf.variable_scope('state_net'):
            # c_names(collections_names) are the collections to store variables
            c_names_state, n_l1_state, w_initializer_state, b_initializer_state = \
                ['state_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 2, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1_state], initializer=w_initializer_state,
                                     collections=c_names_state)
                b1 = tf.get_variable('b1', [1, n_l1_state], initializer=b_initializer_state,
                                     collections=c_names_state)
                l1 = tf.nn.relu(tf.matmul(self.tf_obs2, w1) + b1)
                self.variable_summaries(w1)
                self.variable_summaries(b1)
            tf.summary.histogram('state_net/l1/Wx_plus_b', tf.matmul(self.tf_obs2, w1) + b1)
            tf.summary.histogram('state_net/l1/output', l1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1_state, self.n_state], initializer=w_initializer_state,
                                     collections=c_names_state)
                b2 = tf.get_variable('b2', [1, self.n_state], initializer=b_initializer_state,
                                     collections=c_names_state)
                self.state = tf.matmul(l1, w2) + b2
                self.variable_summaries(w2)
                self.variable_summaries(b2)
            tf.summary.histogram('state_net/l2/state', self.state)

        # ------------------------ build policy_net ------------------------
        with tf.variable_scope('policy_net2'):
            # c_names(collections_names) are the collections to store variables
            c_names_policy, n_l1_policy, w_initializer_policy, b_initializer_policy = \
                ['policy_net2_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_state, n_l1_policy], initializer=w_initializer_policy,
                                     collections=c_names_policy)
                b1 = tf.get_variable('b1', [1, n_l1_policy], initializer=b_initializer_policy,
                                     collections=c_names_policy)
                l1 = tf.nn.tanh(tf.matmul(self.state, w1) + b1)
                self.variable_summaries(w1)
                self.variable_summaries(b1)
            tf.summary.histogram('policy_net2/l1/Wx_plus_b', tf.matmul(self.state, w1) + b1)
            tf.summary.histogram('policy_net2/l1/output', l1)

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
            tf.summary.histogram('policy_net2/l2/act_prob', self.all_act_prob)

        #优化policy网络
        with tf.name_scope('loss_policy'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts2)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt2)  # reward guided loss
        tf.summary.scalar('loss_policy', self.loss)
        with tf.name_scope('train'):
            # 选择待优化的参数为policy_net_params
            optimize_varList_q = tf.get_collection('policy_net2_params')
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=optimize_varList_q)

        #优化state网络
        with tf.name_scope('loss_state'): #使用cross entropy，期望预测值概率分布能够接近真实值概率分布
            self.loss_state = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_acts_star2,
                                                                                     logits=self.all_act))
        tf.summary.scalar('loss_state', self.loss_state)
        with tf.name_scope('train_state'):
            # 选择待优化的参数为state_net_params
            optimize_varList_state = tf.get_collection('state_net_params')
            self._train_op_state = tf.train.AdamOptimizer(self.lr_S).minimize(self.loss_state,
                                                                              var_list=optimize_varList_state)
        #记录reward
        with tf.name_scope('reward'):
            self.sumReward = tf.reduce_sum(self.tf_rs2)
        tf.summary.scalar('reward', self.sumReward)



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
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs2: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, a_star, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_as_star.append(a_star)
        self.ep_rs.append(r)

    def learn(self, step):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _, loss, loss_state, train_summary, value = self.sess.run([self.train_op, self.loss, self.loss_state, self.merged, self.sumReward], feed_dict={
             self.tf_obs2: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts2: np.array(self.ep_as),  # shape=[None, ]
             self.tf_acts_star2: np.array(self.ep_as_star),
             self.tf_vt2: discounted_ep_rs_norm,  # shape=[None, ]
             self.tf_rs2: self.ep_rs,
        })

        self.train_writer.add_summary(train_summary, step)  # 调用train_writer的add_summary方法将训练过程以及训练步数保存

        self.ep_obs, self.ep_as, self.ep_as_star, self.ep_rs = [], [], [], []    # empty episode data
        return discounted_ep_rs_norm, loss, loss_state

    def learn_state(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _, loss_state = self.sess.run(
            [self._train_op_state, self.loss_state], feed_dict={
                self.tf_obs2: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                self.tf_acts2: np.array(self.ep_as),  # shape=[None, ]
                self.tf_acts_star2: np.array(self.ep_as_star),
                self.tf_vt2: discounted_ep_rs_norm,  # shape=[None, ]
            })

        return discounted_ep_rs_norm, loss_state

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



