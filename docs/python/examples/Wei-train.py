import gym
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.tune.logger import pretty_print
from ray.rllib.agents.registry import get_agent_class
import logging


import ptvsd
ptvsd.enable_attach()
print("Waiting for debugger to attach")
ptvsd.wait_for_attach() # (Optional) This blocks until a debugger has attached

# ray.init(num_cpus=4, local_mode=True, redis_address="10.1.3.4:6379")
ray.init(num_cpus=4, local_mode=True, logging_level=logging.DEBUG)

#PPO
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config["num_gpus"] = 0
ppo_config["num_workers"] = 1
cls = get_agent_class("PPO")
# ppo_trainer = cls(config=ppo_config, env="CartPole-v0")
# ppo_trainer = ppo.PPOTrainer(config=ppo_config, env="CartPole-v0")
ppo_trainer = ppo.PPOAgent(config=ppo_config, env="CartPole-v0")

for i in range(3):
    ppo_result = ppo_trainer.train()

env = gym.make("CartPole-v0")
state = env.reset()
done = False
cumulative_reward = 0

import shutil
import os
export_dir = '/home/awoods/exps/exp0'
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
ppo_trainer.export_policy_model(export_dir)

print("Observation:")
print(state)

# full_fetch does not exist in Athens Ray fork, perhaps that is key?
# action, something, states = ppo_trainer.compute_action(state, full_fetch=True)
action = ppo_trainer.compute_action(state)
action, something, states = ppo_trainer.compute_action(state)
state, reward, done, _ = env.step(action)

print("Action:")
print(action)
print("States:")
print(states)
