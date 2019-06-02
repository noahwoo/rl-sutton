import numpy as np

from gym.envs.toy_text import discrete

class GamblerEnv(discrete.DiscreteEnv):

    """
    Gambler environment from Sutton's Reinforcement Learning book chapter 4.
    A gambler make bets on the outcome of a sequence of coin flips, he wins
    the same amount of dollars as the stake if the outcome is head, and lose
    if tail. The game ends if reaching goal of 100$ or running out of money.
    - *state*:   the gambler's capital s \in {1, 2, ..., 99}
    - *actions*: stakes, a \in {0, 1, ..., min(s, 100-s)}
    - *reward*:  zero on all transitions except those on which the gambler
      reaches his goal, when it is +1
    - *policy*:  the mapping from levels of capital to stakes
    - *environment*: defined by p_h, the probability of head coming up in each
      flip

    """

    metadata = {}

    def __init__(self, ph = 0.4):

        self.ph = ph

        nS = 101 # {0, 1, ..., 100}
        nA = 50

        P = {}

        for s in range(nS):
            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(1, min(s, 100-s)+1)}

            is_done = lambda s: s == 0 or s == 100
            reward = 1.0 if is_done(s) and s == 100 else 0.0

            # We're stuck in a terminal state
            if is_done(s):
                for a in range(1, min(s, 100-s)+1):
                    P[s][a] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                for a in range(1, min(s, 100-s)+1):
                    P[s][a] = [(ph, s+a, reward, False), (1-ph, s-a, reward, False)]

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GamblerEnv, self).__init__(nS, nA, P, isd)

    def _render(self, close=False):
        return
