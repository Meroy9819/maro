# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from maro.rl.algorithms.torch.policy_optimization import PolicyOptimization
from maro.rl.utils.trajectory_utils import get_k_step_returns


class PolicyGradient(PolicyOptimization):
    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)  # (N, state_dim)
        returns = get_k_step_returns(reward_sequence, self._reward_decay)
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = self._policy_model(states).gather(1, actions.unsqueeze(1)).squeeze()  # (N, 1)
        policy_loss = -(torch.log(action_prob) * returns).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()
