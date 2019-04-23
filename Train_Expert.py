"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient                 #train
# from RL_brain_test import PolicyGradient          #test

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

#终止奖励
TeminalReward = -50
#终止回合
TeminalOnce = -10000
# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
env = gym.make('Acrobot-v1')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
# print(ca)

RL = PolicyGradient(
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

for i_episode in range(1000):

    observation = env.reset()

    while True:
        # if True: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        ep_rs_sum = sum(RL.ep_rs)

        if done or ep_rs_sum < TeminalOnce:

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            vt, loss = RL.learn(i_episode)
            print("episode:"+str(i_episode)+"  reward:"+str(ep_rs_sum)+"  loss:"+str(loss))

            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_
    if ep_rs_sum > TeminalReward:
        break

RL.saver.save(RL.sess, 'ckpt/model.ckpt')
#########################################  train  #######################################


#########################################  test  #######################################
# for i_episode in range(10):
#
#     observation = env.reset()
#     print(observation)
#     ep_rs = []
#     while True:
#         env.render()
#
#         action = RL.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)
#
#         ep_rs.append(reward)
#         if done:
#             ep_rs_sum = sum(ep_rs)
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