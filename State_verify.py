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
# from RL_brain_test import PolicyGradient
from RL_brain_inverse_test import PolicyGradientInverse
import numpy as np
import tensorflow as tf

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradientInverse(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)
# print(RL.sess.run(tf.get_collection('state_net_params')))
# print(RL.sess.run(tf.get_collection('policy_net2_params')))
# print(ca)


#########################################  verify-state  #######################################

for i_episode in range(1000):

    observation = env.reset()
    count = 0   #计数器
    action_star = [1, 0, 0] #这个不使用
    while True:
        # if True: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, action_star, reward)


        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            vt, loss, loss_state = RL.learn(i_episode)
            print("episode:"+str(i_episode)+"  reward:"+str(ep_rs_sum)+"  loss:"+str(loss)+"  loss_state:"+str(loss_state))

            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_
        count = count + 1

    if ep_rs_sum > -200:
        break
    # if ep_rs_sum > 200000:
    #     break

RL.saver.save(RL.sess,'ckpt/model.ckpt')
#########################################  verify-state  #######################################


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