import os
import numpy as np
from datetime import datetime
import sys

from omegaconf import OmegaConf
import hydra
from datetime import datetime
from omegaconf import DictConfig
import wandb

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch



def train(args):
    # Set up logging using wandb
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    #cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = {'project': 'Gym','group': 'Bruce-P2'}
    wandb_logger = wandb.init(
                            config=cfg_dict, 
                            project='Gym',
                            group='Bruce-P2',
                            name=log_name
                            )
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True, wandb_logger=wandb_logger)

if __name__ == '__main__':
    args = get_args()
    args.headless = False
    train(args)
