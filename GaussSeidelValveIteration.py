'''
@File     : GaussSeidelValveIteration.py
@Copyright: 
@author   : lxt
@Date     : 2020/5/4
@Desc     :
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class GaussSeidelValueIteration():
    def __init__(self, values_shape, qvalues_shape, gamma=0.9, delta = 1e-6, absorbing=True):
        self.values = np.zeros(values_shape)
        self.qvalues = np.zeros(qvalues_shape)
        self.policy = np.ones(qvalues_shape) / 4
        self.gamma = gamma
        self.absorbing = absorbing
        self.delta = delta

    def do_one_step(self):
        oldValues = self.values.copy()
        m, n = self.values.shape
        cur_best = 0
        for i in range(m):
            for j in range(n):
                self.qvalues[i][j][0] = self.compute_q(i, j, 0)
                cur_best = self.qvalues[i][j][0]
                best_action = 0
                for action in range(1, 4):
                    self.qvalues[i][j][action] = self.compute_q(i, j, action)
                    if self.qvalues[i][j][action] > cur_best:
                        cur_best = self.qvalues[i][j][action]
                        best_action = action
                self.policy[i][j] = np.eye(4)[best_action]
                self.values[i][j] = cur_best
        diff = self.values - oldValues
        diff = np.max(np.abs(diff))
        # self.values = newValues
        return diff

    def gauss_seidel_value_iteration(self):
        cur_iterate_times = 0
        start = datetime.now()
        while True:
            print("Iterate No.{0}".format(cur_iterate_times))
            # diff = self.do_one_step()
            diff = self.do_one_step_inverse()
            cur_iterate_times += 1
            if diff < self.delta:
                end = datetime.now()
                print((end - start).total_seconds())
                self.printValue()
                self.print_policy()
                self.show_values()
                break

    def do_one_step_inverse(self):
        oldValues = self.values.copy()
        m, n = self.values.shape
        cur_best = 0
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                self.qvalues[i][j][0] = self.compute_q(i, j, 0)
                cur_best = self.qvalues[i][j][0]
                best_action = 0
                for action in range(1, 4):
                    self.qvalues[i][j][action] = self.compute_q(i, j, action)
                    if self.qvalues[i][j][action] > cur_best:
                        cur_best = self.qvalues[i][j][action]
                        best_action = action
                self.policy[i][j] = np.eye(4)[best_action]
                self.values[i][j] = cur_best
        diff = self.values - oldValues
        diff = np.max(np.abs(diff))
        return diff



    def compute_q(self, i, j, action):
        if i == 7 and j == 8:
            if self.absorbing:
                return 10.0
            else:
                return 10.0 + self.gamma * 0.25 * self.values[0][0] + self.values[0][9] + self.values[9][0] + self.values[9][9]
        if i == 2 and j == 7:
            if self.absorbing:
                return 3.0
            else:
                return 3.0 + self.gamma * 0.25 * self.values[0][0] + self.values[0][9] + self.values[9][0] + self.values[9][9]
        newqval = 0.0
        for dir in range(4):
            contrib = self.contribution(i, j, dir)
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
    def contribution(self, x, y, action):
        if action == 0:
            if x == 0:
                return -1.0 + self.gamma * self.values[x][y]
            else:
                return self.gamma * self.values[x-1][y]
        elif action == 1:
            if y == 9:
                return -1.0 + self.gamma * self.values[x][y]
            else:
                return self.gamma * self.values[x][y+1]
        elif action == 2:
            if x == 9:
                return -1.0 + self.gamma * self.values[x][y]
            else:
                return self.gamma * self.values[x+1][y]
        elif action == 3:
            if y == 0:
                return -1.0 + self.gamma * self.values[x][y]
            else:
                return self.gamma * self.values[x][y-1]
        else:
            return 0

    def printValue(self):
        m, n = self.values.shape
        for i in range(m):
            for j in range(n):
                print('{0:>6.2f}'.format(self.values[i][j]), end=" ")
            print("")
        print()
    def print_policy(self):
        action_dict = ["U", "R", "D", "L"]
        m, n, _ = self.policy.shape
        print("高斯赛德尔值迭代的策略矩阵")
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
        plt.title("gauss_seidel_values_iteration")
        plt.show()

def main():
    for i in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        gsvi = GaussSeidelValueIteration(values_shape=[10, 10], qvalues_shape=[10, 10, 4], gamma=i)
        gsvi.gauss_seidel_value_iteration()

if __name__ == '__main__':
    main()

