import os

from collections import defaultdict, deque, namedtuple
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "../output")
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])

# plot curve
def plot_learning_curve(filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)

class ReplayMemory :
    def __init__(self, capacity = 100000, replace = False) :
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.fields = Transition._fields

    def add(self, record) :
        if isinstance(record, Transition) :
            self.buffer.append(record)
        elif isinstance(record, list) :
            self.buffer += record

        while self.capacity and self.size > self.capacity :
            self.buffer.pop(0)

    def _reformat(self, ids) :
        return {
            field_name: np.array(
                [getattr(self.buffer[id], field_name) for id in ids])
            for field_name in self.fields
        }

    def sample(self, batch_size) :
        assert self.size >= batch_size
        ids = np.random.choice(range(self.size), size=batch_size, replace = self.replace)
        return self._reformat(ids)

    def pop(self, batch_size) :
        pop_size = min(self.size, batch_size)
        batch = self._reformat(range(pop_size))
        self.buffer = self.buffer[pop_size:]
        return batch

    @property
    def size(self) :
        return len(self.buffer)