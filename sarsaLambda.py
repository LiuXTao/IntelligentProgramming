import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class SarsaLambda:
    def __init__(self, play_rounds=1000, gamma=0.9, alpha=0.5, epsilon=0.1, lambdas=0.5):
        self.Qtable, self.Z = self.create_Qtable_Z()
        self.step_record = []
        self.reward_avg = []
        self.reward_record = []
        self.play_rounds = play_rounds
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.gameEnv = None
        self.lambda_ = lambdas


    def create_Qtable_Z(self, rows=4, columns=12):
        # Qtable 为 [action, states_num]
        Qtable = np.zeros((4, columns*rows))
        Z = np.zeros((4, columns*rows))
        # 'up': Qtable[0, :], 'left': Qtable[1, :],
        # 'right':Qtable[2, :], 'down':Qtable[3, :]
        return Qtable, Z


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

    def sarsa_lambda(self):
        for iters in range(self.play_rounds):
            self.Z = np.zeros(np.shape(self.Qtable))
            agent = (3, 0)
            game_over = 0
            reward_sum = 0
            step_count = 0
            # 初始化悬崖棋盘
            env = np.zeros((4, 12))
            env = self.visited_env(agent, env)
            state, _ = self.get_state(agent)
            action = self.epsilon_greedy_policy(state=state)
            while game_over != 1:
                agent = self.move(action=action, agent=agent)
                env = self.visited_env(agent, env)
                step_count += 1
                next_state, _ = self.get_state(agent)
                reward, game_over = self.get_reward(next_state)
                reward_sum += reward
                if game_over == -1:  # 上一步踏进了悬崖, 则返回
                    agent = (3, 0)
                    next_state, _ = self.get_state(agent)
                next_action = self.epsilon_greedy_policy(state=next_state)

                delta = reward + self.gamma * self.Qtable[next_action][next_state] - self.Qtable[action][state]
                self.Z[action][state] += 1
                self.Qtable = self.Qtable + self.alpha * delta * self.Z
                self.Z = self.gamma * self.lambda_ * self.Z
                action, state = next_action, next_state
            self.reward_record.append(reward_sum)
            self.step_record.append(step_count)
            if iters == self.play_rounds - 1:
                print("Agent trained with SARSA("+ str(self.lambda_)+") after", self.play_rounds, " iteration")
                print(env)
                self.gameEnv = env
                self.calculate_avg_reward()

    def calculate_avg_reward(self, length=500):
        data = np.array(self.reward_record)
        result = [np.mean(data[i:i + 20]) for i in range(length)]
        self.reward_avg = result

    def visulize_heatmap(self, show_data, method="Sarsa Lambda"):
        plt.figure()
        data = np.mean(show_data, axis=0)
        data = data.reshape((4, 12))
        ax = sns.heatmap(np.array(data))
        plt.title(method)
        plt.show()

    def visulize_env(self, show_data, method="Sarsa Lambda"):
        plt.figure()
        ax = sns.heatmap(np.array(show_data))
        plt.title(method)
        plt.show()

    def getResult(self):
        return self.Qtable, self.reward_record, self.reward_avg, self.step_record, self.gameEnv
