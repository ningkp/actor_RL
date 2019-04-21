import gym

env = gym.make('Pendulum-v0')

for i_episode in range(100):
    observation = env.reset()
    # print(observation)
    # print(observation.shape)
    # print(env.action_space)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # print(ca)
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation_, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break