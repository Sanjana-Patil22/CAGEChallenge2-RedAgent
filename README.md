# Repository for the advanced computer security class - Winter 2023

The project uses the simulator [CybORG](https://github.com/cage-challenge/cage-challenge-2/tree/main). Please follow the [instructions](https://github.com/cage-challenge/cage-challenge-2/tree/main/CybORG) to install the simulator.

## Requirements

The Cyborg Simulator and pytorch

## Project

The objective is each group gets a red agent that maximizes the reward using reinforcement learning.

In the current version, the red agent produces a random action sampled from the action space. 

The `Agents/RedAgent.py` file implements the class that will contain the policy. 

Currently, this function implements a random policy. Each group should modify the get action function to obtain an action from their red agent. Similarly, each group needs to modify the train function to implement the training function that will return the red agent policy.

Additionally, each group needs to modify the red train.py to call their implemented train function. Finally, the groups can use the red evaluation.py to evaluate their agent’s policy.