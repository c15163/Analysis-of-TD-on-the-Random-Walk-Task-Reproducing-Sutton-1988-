import random as random
import numpy as np
from matplotlib import pyplot as plt


class TDLambdaRandomWalk(object):
    """
    Reproducing Sutton (1988): TD(lambda) on the random walk task.

    - 7 states: A, B, C, D, E, F, G (A,G are terminal)
    - State representation: one-hot vector of length 7
    - Two experiments:
        1) Repeated presentations until weight convergence
        2) Single pass over training sequences (no convergence criterion)
    """

    def __init__(self, state=3):
        # start from state D (index 3) by default
        self.state = state
        # one-hot vectors for 7 states A~G
        self.vectors = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # A
            [0, 1, 0, 0, 0, 0, 0],  # B
            [0, 0, 1, 0, 0, 0, 0],  # C
            [0, 0, 0, 1, 0, 0, 0],  # D
            [0, 0, 0, 0, 1, 0, 0],  # E
            [0, 0, 0, 0, 0, 1, 0],  # F
            [0, 0, 0, 0, 0, 0, 1],  # G
        ], dtype=object)

    def make_sequence(self):
        """
        Generate one random-walk episode from D until reaching A or G.
        Transition probability: 0.5 left, 0.5 right.
        """
        direction = ['left', 'right']
        # D = index 3
        self.state = 3
        x = []
        x.append(self.vectors[3])

        # continue until we hit terminal state A(0) or G(6)
        while 0 < self.state < 6:
            where = random.choice(direction)
            if where == 'left':
                self.state -= 1
                if self.state == 0:
                    x.append(self.vectors[0])
                    break
                elif self.state == 1:
                    x.append(self.vectors[1])
                elif self.state == 2:
                    x.append(self.vectors[2])
                elif self.state == 3:
                    x.append(self.vectors[3])
                elif self.state == 4:
                    x.append(self.vectors[4])
            else:
                self.state += 1
                if self.state == 6:
                    x.append(self.vectors[6])
                    break
                elif self.state == 2:
                    x.append(self.vectors[2])
                elif self.state == 3:
                    x.append(self.vectors[3])
                elif self.state == 4:
                    x.append(self.vectors[4])
                elif self.state == 5:
                    x.append(self.vectors[5])

        return np.array(x)

    def experiment_1(self, alpha, lamda):
        """
        Experiment 1:
        - Repeat presentations until weight vector converges (||w_new - w_old|| < epsilon)
        - Weight updated once per training set (10 sequences)
        - Returns mean RMSE over 100 training sets.
        """
        real_weight = np.array([1/6, 2/6, 3/6, 4/6, 5/6], dtype=object)
        number_of_training = 100
        number_of_sequence = 10
        rmse = []
        epsilon = 1e-3

        for _ in range(number_of_training):
            # weights for A..G (terminal probs fixed at 0 and 1)
            weight = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1], dtype=object)
            prev_w = np.array([0, 0, 0, 0, 0], dtype=object)

            while True:
                prev_w = weight[1:-1]
                over_sequence_delta_weight = np.zeros(7, dtype=object)

                # accumulate delta-weight over all sequences
                for _ in range(number_of_sequence):
                    TD_error = 0
                    x = self.make_sequence()
                    m = np.shape(x)[0]
                    delta_weight = np.zeros(7, dtype=object)
                    prediction = np.dot(x, np.transpose(weight))

                    for k in range(0, m - 1):
                        eligibility = np.zeros(7, dtype=object)
                        TD_error = prediction[k + 1] - prediction[k]
                        for l in range(0, k + 1):
                            eligibility = eligibility + ((lamda ** (k - l)) * x[l])
                        delta_weight = delta_weight + alpha * TD_error * eligibility

                    over_sequence_delta_weight += delta_weight

                # update weight vector
                weight[1:-1] = weight[1:-1] + over_sequence_delta_weight[1:-1]

                # convergence check
                if np.linalg.norm(weight[1:-1] - prev_w, 2) < epsilon:
                    break

            rmse.append(
                np.sqrt(
                    np.sum((real_weight - weight[1:-1]) ** 2) / 5
                )
            )

        return np.mean(rmse)

    def experiment_2(self, alpha, lamda):
        """
        Experiment 2:
        - No convergence criterion
        - Weight updated after each sequence
        - Training set presented only once.
        - Returns mean RMSE over 100 training sets.
        """
        real_weight = np.array([1/6, 2/6, 3/6, 4/6, 5/6], dtype=object)
        number_of_training = 100
        number_of_sequence = 10
        rmse = []

        for _ in range(number_of_training):
            weight = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1], dtype=object)

            for _ in range(number_of_sequence):
                TD_error = 0
                x = self.make_sequence()
                m = np.shape(x)[0]
                delta_weight = np.zeros(7, dtype=object)
                prediction = np.dot(x, np.transpose(weight))

                for k in range(0, m - 1):
                    eligibility = np.zeros(7, dtype=object)
                    TD_error = prediction[k + 1] - prediction[k]
                    for l in range(0, k + 1):
                        eligibility = eligibility + ((lamda ** (k - l)) * x[l])
                    delta_weight = delta_weight + alpha * TD_error * eligibility

                # update after each sequence
                weight[1:-1] = weight[1:-1] + delta_weight[1:-1]

            rmse.append(
                np.sqrt(
                    np.sum((real_weight - weight[1:-1]) ** 2) / 5
                )
            )

        return np.mean(rmse)


def run_td_lambda_experiments():
    # Figure 2: average error vs lambda (repeated presentations)
    alpha = 0.2
    agent3 = TDLambdaRandomWalk()
    lamda_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    error3 = []

    for lam in lamda_list:
        error3.append(agent3.experiment_1(alpha, lam))

    plt.figure()
    plt.title('Figure 2. Average Error under repeated presentations', size=15)
    plt.plot(lamda_list, error3, '-o')
    plt.xlabel("Lambda", size=15)
    plt.ylabel("RMSE", size=15)
    plt.savefig("figure2.png")

    # Figure 3: average error after 10 sequences (lambda, alpha grid)
    error4 = []
    lamda_grid = [0.0, 0.3, 0.8, 1.0]
    alpha_grid = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                  0.4, 0.45, 0.5, 0.55, 0.6]
    agent4 = TDLambdaRandomWalk()

    for lam in lamda_grid:
        for a in alpha_grid:
            error4.append(agent4.experiment_2(a, lam))

    plt.figure()
    plt.title('Figure 3. Average Error after 10 sequences', size=15)
    plt.plot(alpha_grid, error4[0:13], '-o', label='lambda = 0.0')
    plt.plot(alpha_grid, error4[13:26], '-o', label='lambda = 0.3')
    plt.plot(alpha_grid, error4[26:39], '-o', label='lambda = 0.8')
    plt.plot(alpha_grid, error4[39:], '-o', label='lambda = 1.0')
    plt.ylim(0.0, 0.8)
    plt.xlabel("alpha", size=15)
    plt.ylabel("RMSE", size=15)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig("figure3.png")

    # Figure 4: best alpha (all lambdas included)
    error_2d1 = np.zeros((len(lamda_grid), len(alpha_grid)))
    error_2d2 = np.zeros((len(lamda_grid) - 2, len(alpha_grid)))

    for i in range(len(lamda_grid)):
        for j in range(len(alpha_grid)):
            error_2d1[i][j] = error4[i * len(alpha_grid) + j]

    for i in range(len(lamda_grid) - 2):
        for j in range(len(alpha_grid)):
            error_2d2[i][j] = error4[i * len(alpha_grid) + j]

    error_1d_sum1 = np.mean(error_2d1, axis=0)
    error_1d_sum2 = np.mean(error_2d2, axis=0)

    best_alpha1 = alpha_grid[np.argmin(error_1d_sum1)]
    best_alpha2 = alpha_grid[np.argmin(error_1d_sum2)]

    agent5 = TDLambdaRandomWalk()
    lamda_full = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    error5_1 = []
    for lam in lamda_full:
        error5_1.append(agent5.experiment_2(best_alpha1, lam))

    print("best alpha (including Widrow):", best_alpha1)

    plt.figure()
    plt.title('Figure 4. Average Error with best alpha', size=15)
    plt.plot(lamda_full, error5_1, '-o')
    plt.xlabel("Lambda", size=15)
    plt.ylabel("RMSE", size=15)
    plt.savefig("figure4.png")

    # Figure 5: best alpha excluding lambda = 0.8, 1.0 (Widrow case)
    error5_2 = []
    for lam in lamda_full:
        error5_2.append(agent5.experiment_2(best_alpha2, lam))

    print("best alpha (excluding Widrow):", best_alpha2)

    plt.figure()
    plt.title('Figure 5. Average Error with best alpha excluding Widrow', size=15)
    plt.plot(lamda_full, error5_2, '-o')
    plt.xlabel("Lambda", size=15)
    plt.ylabel("RMSE", size=15)
    plt.savefig("figure5.png")


if __name__ == "__main__":
    run_td_lambda_experiments()