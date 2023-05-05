import yaml
import logging
import logging.config

import sys
import random

from torch.autograd import Variable
import torch

class Options:
    def __init__(self, options):
        self.options = options
    def __getattr__(self, attr):
        return self.options[attr]
    __getitem__ = __getattr__
    def __str__(self):
        return str(self.options)
    __repr__ = __str__

def parse_options(file=None):
    if file is None:
        file = 'configs/default_options.yaml'
    with open(file, mode='r') as f:
        options = yaml.safe_load(f)
    return Options(options)


def get_logger(file, stdout=True):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
        logger = logging.getLogger('logger')
        if stdout:
            logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        
    return logger


def logger_test(logger):
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

    
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


if __name__ == '__main__':
    print(parse_options())
    logger_test(get_logger())