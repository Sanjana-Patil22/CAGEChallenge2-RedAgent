# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.WrappedAgent import WrappedBlueAgent
from Agents.MainAgent import MainAgent
from Agents.RedAgent import RedAgent
import random

agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()

    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'


    # Load blue agent
    blue_agent = WrappedBlueAgent
    # Set up environment with blue agent running in the background and
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")

    # Load the red agent from the saved model file created when running red_train.py
    red_agent = RedAgent(env, model_file="models/Scenario2/red_ppo_agent")

    # Settings mentioned by the professor to use as our evaluation parameters.
    num_steps = 30
    max_eps = 100
    observation = env.reset()

    # action_space = wrapped_cyborg.get_action_space(agent_name)
    action_space = env.get_action_space(agent_name)
    total_reward = []
    actions = []
    for i in range(max_eps):
        r = []
        a = []
        for j in range(num_steps):
            action = red_agent.get_action(observation, action_space)
            observation, rew, done, info = env.step(action)
            print(f"Action: {info.get('action')} ({action}), Reward: {rew}")
            r.append(rew)
            a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

        total_reward.append(sum(r))
        actions.append(a)
        observation = env.reset()

        print(f"Average Reward for episode {i}: {sum(r) / num_steps}")

    print(f"Average reward for each episode: {sum(total_reward) / max_eps}")
