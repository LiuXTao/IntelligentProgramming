import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class NStepSarsa:
    def __init__(self, play_rounds=1000, gamma=0.9, alpha=0.5, epsilon=0.1, n=10):
        self.Qtable = self.create_Qtable()
        self.step_record = []
        self.reward_avg = []
        self.reward_record = []
        self.play_rounds = play_rounds
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.gameEnv = None
        self.n = n

    def create_Qtable(self, rows=4, columns=12):
        # Qtable 为 [action, states_num]
        Qtable = np.zeros((4, columns*rows))
        # 'up': Qtable[0, :], 'left': Qtable[1, :],
        # 'right':Qtable[2, :], 'down':Qtable[3, :]
        return Qtable


    # 𝜀 −贪心行动选择行动策略
    def epsilon_greedy_policy(self, state):
        prob = np.random.random()
        if prob < self.epsilon:
            action = np.random.choice(4)
            # print("random", action)
        else:

            action = np.argmax(self.Qtable[:, state])
            # print("max", action)
        return action

    # 根据行动更新座标
    def move(self, agent, action):
        x, y = agent
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y > 0:
            y -= 1
        elif action == 2 and y < 11:
            y += 1
        elif action == 3 and x < 3:
            x += 1
        # 获取下一步的agent坐标
        agent = (x, y)
        return agent

    # 获取当前状态
    def get_state(self, agent):
        x, y = agent
        # (4, 12)， 所以当前格子state计算如 12*x + y
        state = 12 * x + y
        state_action = self.Qtable[:, int(state)]
        max_state = np.amax(state_action)
        return state, max_state

    # 获取当前状态可取的收益
    def get_reward(self, state):
        game_over = 0
        reward = -1
        # 终点
        if state == 47:
            game_over = 1
            reward = 0
        #  如果在悬崖边上
        if state >= 37 and state < 47:
            game_over=-1
            reward = -100
        return reward, game_over

    def visited_env(self, agent, env):
        x, y = agent
        env[x][y] = 1
        return env

    def nstep_sarsa(self):
        # total_reward = 0
        for iters in range(self.play_rounds):
            agent = (3, 0)
            game_over = 0
            reward_sum = 0
            # 初始化悬崖棋盘
            env = np.zeros((4, 12))
            env = self.visited_env(agent, env)
            T = sys.maxsize
            tau = 0
            t = -1
            buffer_s, buffer_a, buffer_r = [], [], []
            state, _ = self.get_state(agent)
            action = self.epsilon_greedy_policy(state)
            buffer_a.append(action)
            buffer_s.append(state)

            while tau < T -1:
                t += 1
                if t < T:
                    agent = self.move(action=action, agent=agent)
                    env = self.visited_env(agent, env)
                    next_state, _ = self.get_state(agent)
                    reward, game_over = self.get_reward(next_state)
                    reward_sum += reward
                    buffer_r.append(reward)
                    buffer_s.append(next_state)

                    if game_over == 1:
                        T = t+1
                    else:
                        if game_over == -1: # 上一步踏进了悬崖, 则返回
                            agent = (3, 0)
                            next_state, _ = self.get_state(agent)
                        next_action = self.epsilon_greedy_policy(state=next_state)
                        buffer_a.append(next_action)
                        action = next_action
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau+self.n, T) +1):
                        G += self.gamma**(i - tau - 1) * buffer_r[i-1]
                    if tau + self.n < T:
                        G += self.gamma**self.n * self.Qtable[buffer_a[tau+self.n]][buffer_s[tau+self.n]]
                    self.Qtable[buffer_a[tau]][buffer_s[tau]] += self.alpha * \
                                                                 (G - self.Qtable[buffer_a[tau]][buffer_s[tau]])

            self.reward_record.append(reward_sum)
            if iters == self.play_rounds -1:
                print("Agent trained with "+ str(self.n) + "-step SARSA after" , self.play_rounds , " iteration")
                print(env)
                self.gameEnv = env
                self.calculate_avg_reward()

    def calculate_avg_reward(self, length=500):
        data = np.array(self.reward_record)
        result = [np.mean(data[i:i + 20]) for i in range(length)]
        self.reward_avg = result

    def visulize_heatmap(self, show_data, method="N-step Sarsa"):
        plt.figure()
        data = np.mean(show_data, axis=0)
        data = data.reshape((4, 12))
        ax = sns.heatmap(np.array(data))
        plt.title(method)
        plt.show()

    def visulize_env(self, show_data, method="N-step Sarsa"):
        plt.figure()
        ax = sns.heatmap(np.array(show_data))
        plt.title(method)
        plt.show()

    def getResult(self):
        return self.Qtable, self.reward_record, self.reward_avg, self.step_record, self.gameEnv

