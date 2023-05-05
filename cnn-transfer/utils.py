import yaml
import logging
import logging.config

import sys
import random
import os

import torch

class Options:
    def __init__(self, options):
        self.options = {k: Options(v) if isinstance(v, dict) else v
                        for k, v in options.items()}
        self.options[None] = options
    def __getattr__(self, attr):
        return self.options[attr]
    __getitem__ = __getattr__
    def __str__(self):
        return str(self.options)
    __repr__ = __str__

def parse_options(file=None):
    if file is None:
        file = 'default_options.yaml'
    with open(file, mode='r') as f:
        options = yaml.safe_load(f)
    return Options(options)


def clear_log(log_dir):
    for file in os.listdir(log_dir):
        path = os.path.join(log_dir, file)
        if os.path.isfile(path):
            os.remove(path)

def init_dir(assets, name, suffix=None):
    path = os.path.join(assets, name)
    result = [path]
    if not os.path.isdir(path):
        os.mkdir(path)
    for dir_name in (f'{d}-{suffix}' if suffix else d
                     for d in ['log', 'output']):
        dir_path = os.path.join(path, dir_name)
        result.append(dir_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    return result


if __name__ == '__main__':
    init_dir()