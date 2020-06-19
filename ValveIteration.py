'''
@File     : VI.py
@Copyright: 
@author   : lxt
@Date     : 2020/5/4
@Desc     :
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import  datetime
class ValueIteration():
    def __init__(self, values_shape, qvalues_shape, gamma=0.9, delta=1e-6, absorbing=True):
        self.values = np.zeros(values_shape)
        self.qvalues = np.zeros(qvalues_shape)
        self.policy = np.ones(qvalues_shape) / 4
        self.gamma = gamma
        self.absorbing = absorbing
        self.delta = delta

    def do_one_step(self):
        newValues = np.zeros(self.values.shape)
        m, n = self.values.shape
        for i in range(m):
            for j in range(n):
                self.qvalues[i][j][0] = self.compute_q(i, j, 0)
                newValues[i][j] = self.qvalues[i][j][0]
                best_action = 0
                for action in range(1, 4):
                    self.qvalues[i][j][action] = self.compute_q(i, j, action)
                    if self.qvalues[i][j][action] > newValues[i][j]:
                        newValues[i][j] = self.qvalues[i][j][action]
                        best_action = action
                self.policy[i][j] = np.eye(4)[best_action]

        diff = newValues - self.values
        diff = np.max(np.abs(diff))
        self.values = newValues
        return diff

    def value_iteration(self):
        cur_iterate_times = 0
        start = datetime.now()
        while True:
            print("Iterate No.{0}".format(cur_iterate_times))
            diff = self.do_one_step()

            cur_iterate_times += 1
            if diff < self.delta:
                end = datetime.now()
                print((end - start).total_seconds())
                self.print_value()
                self.print_policy()
                self.show_values()
                break


    def compute_q(self, i, j, action):
        # 返回吸收格子(8, 9)的奖赏
        if i == 7 and j == 8:
            if self.absorbing:  # 设置为true表示为吸收格子
                return 10.0
            else:
                return 10.0 + self.gamma * 0.25 * self.values[0][0] + self.values[0][9] + \
                       self.values[9][0] + self.values[9][9]
        # 返回吸收格子(3, 8)的奖赏
        if i == 2 and j == 7:
            if self.absorbing:
                return 3.0
            else:
                return 3.0 + self.gamma * 0.25 * self.values[0][0] + self.values[0][9] + \
                       self.values[9][0] + self.values[9][9]
        newqval = 0.0
        for dir in range(4):
            contrib = self.contribution(i, j, dir)
            if action == dir:  # 指定方向移动的情况，乘以概率0.7
                newqval += 0.7 * contrib
            else:  # 其他方向的情况，乘以概率0.1
                newqval += 0.1 * contrib
        # 表示格子(5,4)的奖赏
        if i == 4 and j == 3:
            newqval += -5.0
        # 表示格子(8,4)的奖赏
        if i == 7 and j == 3:
            newqval += -10.0
        return newqval


    def contribution(self, x, y, action):
        # action: 0上 1右 2下 3左
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
        print("值迭代的策略矩阵")
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
        plt.title("values_iteration")
        plt.show()

def main():
    for i in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        vi = ValueIteration(values_shape=[10, 10], qvalues_shape=[10, 10, 4], gamma=i)
        vi.value_iteration()

if __name__ == '__main__':
    main()

