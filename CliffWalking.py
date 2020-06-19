'''
@File     : CliffWalking.py
@Copyright: 
@author   : lxt
@Date     : 2020/6/3
@Desc     :
'''

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sarsa import Sarsa
from qlearning import QLearning
from nStepSarsa import NStepSarsa
from sarsaLambda import SarsaLambda


def plot_reward(reward_record_qlearning, reward_record_SARSA, n1="q_learning", n2="SARSA"):
    x = range(500)
    plt.plot()
    plt.plot(x, reward_record_qlearning, label=n1)
    plt.plot(x, reward_record_SARSA, label=n2)
    plt.ylabel('Sum of reward during episode', fontsize=18)
    plt.xlabel('Episodes ', fontsize=18)
    plt.title(n1 + "/" + n2 + " Reward plot", fontsize=18)
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def plot_reward_more(reward_record, name=[], title=''):
    x = range(500)
    plt.plot()
    for i in range(len(name)):
        plt.plot(x, reward_record[i], label=name[i])
    plt.ylabel('Sum of reward during episode', fontsize=18)
    plt.xlabel('Episodes ', fontsize=18)
    if len(name) != 1:
        plt.ylim(-600, 0)
    plt.title(title, fontsize=18)
    plt.legend(loc='best', mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    # 实验1
    sarsa = Sarsa(play_rounds=1000, gamma=0.9, alpha=0.2, epsilon=0.1)
    sarsa.sarsa()
    _, _, reward_record_s, _, _ = sarsa.getResult()
    qLearning = QLearning(play_rounds=1000, gamma=0.9, alpha=0.1, epsilon=0.1)
    qLearning.qlearning()
    _, _, reward_record_q, _, _ = qLearning.getResult()
    plot_reward(reward_record_q, reward_record_s)

    # 实验2
    reward_record_ns = []
    ns_name = []
    reward_record_sl = []
    ls_name = []
    all_name = []
    all_reward = []
    for i in [1, 3, 5]:
        nstepSarsa = NStepSarsa(play_rounds=1000, gamma=0.99, alpha=0.01, epsilon=0.1, n=i)
        nstepSarsa.nstep_sarsa()
        _, _, reward_record, _, _ = nstepSarsa.getResult()
        reward_record_ns.append(reward_record)
        ns_name.append(str(i) + "StepSarsa")

    plot_reward_more(reward_record_ns, ns_name, title="NStepSarsa Reward Plot")

    for i in [0, 0.5, 1]:
        sarsaLambda = SarsaLambda(play_rounds=1000, gamma=0.9, alpha=0.1, epsilon=0.1, lambdas=i)
        sarsaLambda.sarsa_lambda()
        _, _, reward_record, _, _ = sarsaLambda.getResult()
        if i == 1:
            continue
        reward_record_sl.append(reward_record)
        ls_name.append("Sarsa("+ str(i)+")")
    plot_reward_more(reward_record_sl, ls_name, title="Sarsa(λ) Reward Plot")
    # 单独显示Sarsa(1)

    plot_reward_more([reward_record], name=["Sarsa(1)"], title="Sarsa(1) Reward Plot")

    # 显示全部的
    reward_record_sl.extend(reward_record_ns)
    ls_name.extend(ns_name)
    plot_reward_more(reward_record_sl, name=ls_name, title="Sarsa(λ)/NStepSarsa Reward Plot")