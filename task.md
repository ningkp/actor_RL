#2019-4-21
目前已经测试过state learning在actor网络中有作用，现阶段工作为丰富实验。

ALL Tasks:

    1. 在不同的强化学习网络上进行state learning，例如A3C、DQN、DDPG等。
    2. 将1在不同的数据集上进行实验，实验内容为迭代次数的形式，比较收敛情况。
    3. 进行例如Inverse reinforcement learning的实验，横坐标为范例样本轨迹数(范例样本个数)，纵坐标为性能指标。
    4*. 若在Actor网络上学习到的状态表述迁移到其它网络上仍能提升效果就更能说明状态学习的有用性！
    
Current task:
    
    2. 在不同数据集上进行实验
        #离散动作
        A. CartPole
        B. MountainCar
        C. GridWorld
        D. Acrobot
        #E. Breakout
        #连续动作
        A. MountainCarContinuous
        B. Pendulum
        
        
    4. 在DQN上测试在Actor网络状态学习后迁移后的结果
    
    
FrameWork:
    
    1. Train_Expert: 训练一个专家，让不加状态表示的强化学习任务跑完到指定的指标。
        A. Train_Expert.py: 训练专家(强化学习)的代码
        B. RL_brain.py: PolicyGradient网络
        
    2. State_Learning: 学习Ws和PI。
        A. State_Learning.py: 训练State + RL的代码
        B. RL_brain_test.py: 调用第一步学出的Expert来辅助第二步的State Learning
        C. RL_brain_inverse.py: State + RL网络
        
    3. State_Verify: 验证State的学习效果如何。
        A. State_verify: 验证State网络学习效果的代码。
        B. RL_brain_inverse_test.py: 调用第二步学出来的State网络迁移到随机初始化的RL网络上进行学习。
    
    
Parameter:
    
    1. MountainCar: Action -> 3, n_state = 1, state_net_input = 2, Teminal reward > -200
    2. CartPole: Action -> 2, n_state = 6, state_net_input = 8, Teminal reward > 200000