import math
import numpy as np
import matplotlib.pyplot as plt
from dist_gen import no_collision_rewards, collision_rewards

communication_network = []


def ec3(T, K, M, sigma):
    Kappa_p = list(range(K))
    A = []
    R = []
    time = [0]*M
    p = 0
    Mp = M
    mu_caps = [[None]*K for _ in range(M)]
    exploration_sequences = []
    Tpi = [0]*M
    Tp = [0]*M

    for m in range(M):
        exploration_sequences.append(list(range(m, K)) + list(range(0, m)))

    while Mp != 0:
        for m in range(Mp):
            mu_caps[m], time[m] = exploration(
                m,
                exploration_sequences[m],
                sigma,
                mu_caps[m],
                T,
                time[m],
                p,
                A
            )
            Lp, BTp, Tpi[m], Tp[m] = communication_calculations(
                Tpi[m], Tp[m], m, sigma, mu_caps[m], A, R, M, T, p, Mp)

        for m in range(1, Mp):
            follower_first(Kappa_p, mu_caps[m], m, Lp, Tpi)

        Mp_new, Kappa_p_new, exploration_sequences[0] = leader(
            M, Mp, Lp, Kappa_p, mu_caps[0], A, R, Tpi, Tp[0], BTp)
        for m in range(1, Mp):
            exploration_sequences[m] = follower_second(m, M, K, A, R, Kappa_p)
        Mp = Mp_new
        Kappa_p = Kappa_p_new
        p += 1

    for m in range(M):
        exploitation(exploration_sequences[m], T, time[m], m)


def exploration(m, exploration_sequence, sigma, mu_cap, T, t, p, A):
    for arm in exploration_sequence:
        reward = 0
        for _ in range((2**p)*math.ceil((sigma**2)*math.log(T, 10))):
            arm_reward = pull_arm(arm)
            reward += arm_reward
            while len(total_rewards) <= t:
                total_rewards.append(0)

            total_rewards[t] += arm_reward
            t += 1

        mu_cap[arm] = reward / \
            ((2**p)*math.ceil((sigma**2)*math.log(T, 10)))

    return mu_cap, t


def communication_calculations(Tpi, Tp, m, sigma, mu_cap, A, R, M, T, p, Mp):
    Tpi = Tpi + (2**p)*math.ceil((sigma**2)*math.log(T, 10))
    Tp = Tp + Mp*(2**p)*math.ceil((sigma**2)*math.log(T, 10))
    BTp = ((2*(sigma**2)*math.log(T, 10)) / Tp)**0.5
    Qp = math.ceil(math.log(1 / BTp, 2))
    Lp = 1 + Qp

    return Lp, BTp, Tpi, Tp


def follower_first(Kappa_p, mu_cap, m, Lp, Tpi):
    mu_bar = mu_cap.copy()

    for k in range(K):
        send(encode(mu_bar[k]))


def follower_second(m, M, K, A, R, Kappa_p):
    A, R, Kappa_p = decode(receive())

    if M - m <= len(A):
        exploration_sequence = [A[M - m - 1]]
    else:
        exploration_sequence = []
        for i in range(m, len(Kappa_p)):
            exploration_sequence.append(Kappa_p[i])

        for i in range(0, m):
            exploration_sequence.append(Kappa_p[i])

    return exploration_sequence


def exploitation(exploration_sequence, T, t, m):
    print(f'Player {m} accepted arms: {exploration_sequence[0]}')
    print(t)
    print(T)
    for i in range(T-t):
        # arm_reward = pull_arm(exploration_sequence[0])
        arm_reward = no_collision_mean_rewards[exploration_sequence[0]]
        while len(total_rewards) <= t+i:
            total_rewards.append(0)
        total_rewards[t+i] += arm_reward


def leader(M, Mp, Lp, Kappa_p, mu_cap, A, R, Tpi, Tp, BTp):
    mu_bar = mu_cap.copy()

    mu_bar_ip = [mu_bar]
    for i in range(1, Mp):
        mu_bar_ip.append([])
        for _ in range(K):
            mu_bar_ip[i].append(decode(receive()))

    mu_bar_p = [sum([mu_bar_ip[i][k]*(Tpi[i]/Tp) for i in range(Mp)])
                for k in range(K)]

    Rp = [k for k in Kappa_p if len(
        [j for j in Kappa_p if mu_bar_p[j] - 2*BTp >= mu_bar_p[k] + 2*BTp]) >= Mp]

    Ap = [k for k in Kappa_p if len(
        [j for j in Kappa_p if mu_bar_p[k] - 2*BTp >= mu_bar_p[j] + 2*BTp]) >= len(Kappa_p) - Mp]

    A += Ap
    R += Rp

    Kappa_p = sorted([k for k in Kappa_p if k not in Ap and k not in Rp])

    for i in range(1, Mp):
        send(encode((A, R, Kappa_p)))

    if M <= len(A):
        Mp = M - len(A)
        exploration_sequence = [A[M-1]]
    else:
        Mp = M - len(A)
        exploration_sequence = Kappa_p.copy()

    return Mp, Kappa_p, exploration_sequence


def decimal_converter(num):
    while num > 1:
        num /= 10
    return num


def decode(x):
    return x


def encode(x):
    return x


def send(value):
    communication_network.append(value)


def receive():
    return communication_network.pop(0)


def pull_arm(arm):
    return next(no_collision_rewards(arm))


if __name__ == "__main__":
    M = 5
    K = 10
    T = int(2*1e4)
    sigma = 0.2

    total_rewards = []

    no_collision_mean_rewards = np.linspace(0.3, 0.84, K)
    ec3(T, K, M, sigma)

    mu_sum = sum([no_collision_mean_rewards[i] for i in range(K-M, K)])
    regret = [mu_sum - time_reward
              for time_reward in total_rewards]

    for i in range(1, len(regret)):
        regret[i] += regret[i-1]

    plt.plot(range(len(total_rewards)), regret)
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.show()
