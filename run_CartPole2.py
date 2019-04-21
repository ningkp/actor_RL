"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
# from RL_brain import PolicyGradient
from RL_brain_test import PolicyGradient
from RL_brain_inverse import PolicyGradientInverse
import numpy as np
import tensorflow as tf

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

#定义老师的evn
# env_tea = gym.make('CartPole-v0')
env_tea = gym.make('MountainCar-v0')
env_tea.seed(1)
env_tea = env_tea.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# #初始化老师
RL_star = PolicyGradient(
    n_actions=env_tea.action_space.n,
    n_features=env_tea.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

RL = PolicyGradientInverse(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)
# print(RL_star.sess.run(tf.get_collection('policy_net_params')))
# print(RL.sess.run(tf.get_collection('policy_net2_params')))
# print(ca)
#########################################  train  #######################################

# for i_episode in range(3000):
#
#     observation = env.reset()
#
#     while True:
#         if True: env.render()
#
#         action = RL.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)
#
#         # RL.store_transition(observation, action, reward)
#
#         if done:
#             ep_rs_sum = sum(RL.ep_rs)
#
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#
#             # vt, loss = RL.learn(i_episode)
#             # print("episode:"+str(i_episode)+"  reward:"+str(ep_rs_sum)+"  loss:"+str(loss))
#
#             # if i_episode == 0:
#             #     plt.plot(vt)    # plot the episode vt
#             #     plt.xlabel('episode steps')
#             #     plt.ylabel('normalized state-action value')
#             #     plt.show()
#             break
#
#         observation = observation_
#     # if ep_rs_sum > -200:
#     #     break
#
# RL.saver.save(RL.sess, 'ckpt/model.ckpt')
#########################################  train  #######################################



#########################################  train-inverse  #######################################
#先让老师跑m次得到范例轨迹数据D
M = 1
for i in range(0, M):
    observation_tea = env_tea.reset()
    count = 0
    while True:
        # env_tea.render()
        action_tea = RL_star.choose_action(observation_tea)
        observation_tea_, _, done_tea, _ = env_tea.step(action_tea)
        action_star = np.zeros(3)
        action_star[action_tea] = 1

        #存储
        RL.store_transition_tea(observation_tea, action_star)
        if done_tea or count >= 1000:
            break
        observation_tea = observation_tea_
        # print(RL.ep_obs_tea)
        count += 1

# print(RL.ep_obs_tea)
# print(np.array(RL.ep_obs_tea).shape)
# print(np.vstack(RL.ep_obs_tea))
# print(ca)
# print(RL.ep_as_star_tea)
# print(RL.ep_as_star_tea.shape)
print("获取"+str(M)+"条范例轨迹数据完成，共"+str(len(RL.ep_as_star_tea))+"个数据")

for i_episode in range(150):

    observation = env.reset()
    # observation_tea = env_tea.reset()
    count = 0   #计数器
    loss_state = 0
    while True:
        # if True: env.render()

        action = RL.choose_action(observation)

        #在老师状态下得到    pi的action和老师的action
        # action_pi = RL.choose_action(observation_tea)
        # action_tea = RL_star.choose_action(observation_tea)
        action_star = np.zeros(3)
        # action_star[action_tea] = 1

        observation_, reward, done, info = env.step(action)
        # observation_tea_, _, done_tea, _ = env_tea.step(action_tea)
        # print("action:"+str(action)+"  best_action:"+str(action_star))

        #存储优化policy网络
        RL.store_transition(observation, action, action_star, reward)

        #如果teacher没有玩完一轮才学习
        # if not done_tea:
        #     #存储优化state网络
        #     RL.store_transition_tea(observation_tea, action_pi, action_star)
        if count >= 10:
            loss_state += RL.learn_state()

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            vt, loss = RL.learn(i_episode)
            print("episode:"+str(i_episode)+"  reward:"+str(ep_rs_sum)+"  loss:"+str(loss)+"  loss_state:"+str(loss_state/(count-10)))

            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_
        # observation_tea = observation_tea_
        count = count + 1

    if ep_rs_sum > -200:
        break

RL.saver.save(RL.sess,'ckpt/model.ckpt')
#########################################  train-inverse  #######################################



#########################################  test  #######################################
# for i_episode in range(10):
#
#     observation = env.reset()
#
#     while True:
#         env.render()
#
#         action = RL.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)
#
#         RL.store_transition(observation, action, reward)
#
#         if done:
#             ep_rs_sum = sum(RL.ep_rs)
#
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#
#             print("episode:"+str(i_episode)+"  reward:"+str(ep_rs_sum))
#
#             # if i_episode == 0:
#             #     plt.plot(vt)    # plot the episode vt
#             #     plt.xlabel('episode steps')
#             #     plt.ylabel('normalized state-action value')
#             #     plt.show()
#             break
#
#         observation = observation_

#########################################  test  #######################################