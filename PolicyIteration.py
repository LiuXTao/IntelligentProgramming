'''
@File     : PolicyIteration.py
@Copyright: 
@author   : lxt
@Date     : 2020/5/4
@Desc     :
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

class PolicyIteration():
    def __init__(self, values_shape, pvalues_shape, gamma=0.9, delta=1e-6, absorbing=True):
        self.values = np.zeros(values_shape)
        self.policy = np.ones(pvalues_shape) / 4
        self.gamma = gamma
        self.absorbing = absorbing
        self.delta = delta

    def policy_eval(self):
        m, n = self.values.shape
        V = np.zeros([m, n])
        while True:
            diff = 0
            for i in range(m):
                for j in range(n):
                    v = 0
                    for a, action_prob in enumerate(self.policy[i][j]):
                        v += action_prob * self.compute_q(V, i, j, a)
                    diff = max(diff, np.abs(v - V[i][j]))
                    V[i][j] = v
            if diff < self.delta:
                break
        return np.array(V)

    def policy_iteration(self):
        m, n = self.values.shape
        cur_iterate_times = 0
        start = datetime.now()
        while True:
            print("Iterate No.{0}".format(cur_iterate_times))
            # 评估当前策略
            self.values = self.policy_eval()
            # 策略改进部分
            policy_stable = True
            for i in range(m):
                for j in range(n):
                    old_action = np.argmax(self.policy[i][j])
                    # 往后看一步找最好的action，如果有多个最好的，随便选一个
                    action_values = np.zeros(4)
                    for a in range(4):
                        action_values[a] = self.compute_q(self.values, i, j, a)
                    best_action = np.argmax(action_values)

                    if old_action != best_action:
                        policy_stable = False
                    self.policy[i][j] = np.eye(4)[best_action]
            cur_iterate_times += 1


            if policy_stable:
                end = datetime.now()
                print((end - start).total_seconds())
                self.print_value()
                self.print_policy()
                self.show_values()
                break



    def compute_q(self, V, i, j, action):
        if i == 7 and j == 8:
            if self.absorbing:
                return 10.0
            else:
                return 10.0 + self.gamma * 0.25 * V[0][0] + V[0][9] + V[9][0] + V[9][9]
        if i == 2 and j == 7:
            if self.absorbing:
                return 3.0
            else:
                return 3.0 + self.gamma * 0.25 * V[0][0] + V[0][9] + V[9][0] + V[9][9]

        newqval = 0.0
        for dir in range(4):
            contrib = self.contribution(V, i, j, dir)
            if action == dir:
                newqval += 0.7 * contrib
            else:
                newqval += 0.1 * contrib

        if i == 4 and j == 3:
            newqval += -5.0
        elif i == 7 and j == 3:
            newqval += -10.0
        return newqval

    # action: 0上 1右 2下 3左
    def contribution(self, V, x, y, action):
        if action == 0:
            if x == 0:
                return -1.0 + self.gamma * V[x][y]
            else:
                return self.gamma * V[x - 1][y]
        elif action == 1:
            if y == 9:
                return -1.0 + self.gamma * V[x][y]
            else:
                return self.gamma * V[x][y + 1]
        elif action == 2:
            if x == 9:
                return -1.0 + self.gamma * V[x][y]
            else:
                return self.gamma * V[x + 1][y]
        elif action == 3:
            if y == 0:
                return -1.0 + self.gamma * V[x][y]
            else:
                return self.gamma * V[x][y - 1]
        else:
            return 0

    def print_value(self):
        m, n = self.values.shape
        for i in range(m):
            for j in range(n):
                print('{0:>6.2f}'.format(self.values[i][j]), end=" ")
            print("")
        print()
    def print_policy(self):
        action_dict = ["U", "R", "D", "L"]
        m, n, _ = self.policy.shape
        print("策略迭代的策略矩阵")
        for i in range(m):
            for j in range(n):
                if (i == 7 and j == 8) or (i==2 and j==7):
                    print('{}'.format(" "), end=" ")
                else:
                    print('{}'.format(action_dict[np.argmax(self.policy[i][j])]), end=" ")
            print("")
        print()

    def show_values(self):
        plt.figure()
        sns.heatmap(self.values[0:, 0:], annot=True, fmt='.2f', cmap='RdGy')
        plt.title("policy_iteration")
        plt.show()

def main():
    for i in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        pi = PolicyIteration(values_shape=[10, 10], pvalues_shape=[10, 10, 4], delta=1e-6,gamma=i)
        pi.policy_iteration()
    # pi.print_policy()

if __name__ == '__main__':
    main()