
""" Implementation of the n-armed bandit problem.
"""


import numpy as np
import matplotlib.pyplot as plt


class ArmedBandit():
    def __init__(self, eta, num_actions):
        self.eta = eta
        self.num_actions = num_actions
        self.average_reward = 0.0
        self.iteration = 0


        self.optimality_history = []
        self.action_history = []
        self.reward_history = []
        self.action_reward_history = [[] for i in range(num_actions)]
        self.cum_reward_history = []

        # The agent's estimate of the action values Q(a) are initialized to 0.
        self.estimate_action_values = np.zeros(num_actions)

        # The optimal action values are selected according to a noraml gaussian
        # distribution with mean 0 and variance 1.
        self.optimal_action_values = np.random.normal(0.0, 1.0, num_actions)

    def get_reward(self, action):
        """ Gets the reward from the environment if the current action is taken.
            The reward function is the actual optimal action value plus noise
            term with mean 0 and variance 1.
        """
        error = np.random.normal(0.0, 1.0, 1)
        return self.optimal_action_values[action] + error[0]

    def run(self, iterations):
        for i in range(iterations):
            self.run_once()
    
    def run_once(self):
        """ Runs the agent once allowing it to collect a reward.
        """
        action = None

        # Throw a die with probability eta and see if we will explore on this
        # iteration or take the optimal action. 
        flip = np.random.random()
        if flip < self.eta:
            action = int(round(np.random.random()*(self.num_actions - 1)))
        else:
            action = np.argmax(self.estimate_action_values)

        # Get the current reward.
        reward = self.get_reward(action)

        # Update the histories.
        if action == np.argmax(self.optimal_action_values):
            self.optimality_history.append(1)
        else:
            self.optimality_history.append(0)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_reward_history[action].append(reward)
        self.cum_reward_history.append(np.mean(self.reward_history))

        # Update the estimated action values.
        self.estimate_action_values[action] = np.mean(self.action_reward_history[action])

        # Increment time.
        self.iteration += 1

    def get_optimality_history(self):
        return self.optimality_history

    def get_iteration(self):
        return self.iteration

    def get_cum_reward_history(self):
        return self.cum_reward_history


def plot_average_reward_vs_iteration(bandit, actions):
    iteration = bandit.get_iteration()
    cum_reward_history = bandit.get_cum_reward_history()
    plt.plot(range(iteration), cum_reward_history)
    plt.title(r'Armed Bandit (%i Actions)' % actions)
    plt.ylabel(r'Average Reward')
    plt.xlabel(r'Iteration')
    plt.show()


def plot_average_reward_vs_iteration_all(bandit_etas, trials, actions, iterations):
    """ Given a list of bandit trials with different exploration etas.
    """
    (fig, ax) = plt.subplots()
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(bandit_etas))))
    for bandit_eta in bandit_etas:
        c = next(color)
        label = 'Eta: %.2f' % bandit_eta[0]
        ax.plot(range(iterations), bandit_eta[1], c=c, label=label)
        ax.legend(loc='lower right', shadow=True)

    plt.title(r'Armed Bandit (%i Actions and %i Trials)' % (actions, trials))
    plt.ylabel(r'Average Reward')
    plt.xlabel(r'Iteration')
    plt.show()


def plot_percent_optimal_actions(bandit, actions):
    iteration = bandit.get_iteration()
    optimality_history = np.array(bandit.get_optimality_history(), dtype=float)
    np.cumsum(optimality_history, out=optimality_history)

    np.divide(optimality_history.astype('float'), np.arange(1, iteration + 1, dtype=float), out=optimality_history)
    plt.plot(range(iteration), optimality_history)
    plt.title(r'Armed Bandit (%i Actions)' % actions)
    plt.ylabel(r'Percent Optimal')
    plt.xlabel(r'Iteration')
    plt.show()



def plot_percent_optimal_actions_all(bandit_etas, trials, actions, iterations):
    (fig, ax) = plt.subplots()
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(bandit_etas))))
    for bandit_eta in bandit_etas:
        c = next(color)
        label = 'Eta: %.2f' % bandit_eta[0]
        ax.plot(range(iterations), bandit_eta[1], c=c, label=label)
        ax.legend(loc='lower right', shadow=True)

    plt.title(r'Armed Bandit (%i Actions and %i Trials)' % (actions, trials))
    plt.ylabel(r'Average Reward')
    plt.xlabel(r'Iteration')
    plt.show()


def average_bandit_trials_reward_histories(bandits):
    # Assume each bandit has been run the same number of iterations.
    iteration = bandits[0].get_iteration()
    cum_reward_histories = []
    for bandit in bandits:
        cum_reward_histories.append(bandit.get_cum_reward_history())
    return (bandits[0].eta, np.mean(np.array(cum_reward_histories), axis=0))


def average_bandit_trials_optimality_histories(bandits):
    # Assume each bandit has been run the same number of iterations.
    iteration = bandits[0].get_iteration()
    optimality_histories = []
    for bandit in bandits:
        optimality_histories.append(bandit.get_optimality_history())
    return (bandits[0].eta, np.mean(np.array(optimality_histories), axis=0))


def generate_bandits(trials, iterations, actions, eta):
    bandits = []
    for i in range(trials):
        print('Trial: ' + str(i))
        ab = ArmedBandit(eta, actions)
        ab.run(iterations)
        bandits.append(ab)
    return bandits


def main():
    # Plot options
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    actions = 10
    trials = 2000
    iterations = 1000
    etas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

    bandits = []
    bandit_etas_rewards = []
    bandit_etas_optimal = []
    for eta in etas:
        bandits = generate_bandits(trials, iterations, actions, eta)
        bandit_etas_rewards.append(average_bandit_trials_reward_histories(bandits))
        bandit_etas_optimal.append(average_bandit_trials_optimality_histories(bandits))
    plot_average_reward_vs_iteration_all(bandit_etas_rewards, trials, actions, iterations)
    plot_percent_optimal_actions_all(bandit_etas_optimal, trials, actions, iterations)


if __name__ == '__main__':
    import pdb
    main()
