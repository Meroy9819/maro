# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.torch.policy_optimization import PolicyOptimization
from maro.rl.utils.trajectory_utils import get_k_step_returns


class ProximalPolicyOptimization(PolicyOptimization):
    """Proximal policy optimization (PPO) algorithm.

    The policy gradient algorithm base on the policy gradient theorem, a.k.a. REINFORCE.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        optimizer_cls: torch optimizer class for the policy model.
        optimizer_params: parameters required for the policy optimizer class.
        clip_ratio (float): clip ratio as defined in PPO's objective function.
        num_training_iterations (int): number of gradient descent steps per call to the ``train`` method.
    """

    def __init__(self, policy_model: nn.Module, optimizer_cls, optimizer_params, num_actions: int,
                 reward_decay: float, clip_ratio: float, num_training_iterations: int):
        super().__init__(policy_model, optimizer_cls, optimizer_params, num_actions, reward_decay)
        self._clip_ratio = clip_ratio
        self._num_training_iterations = num_training_iterations

    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, log_action_prob_sequence: np.ndarray,
              reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        returns = get_k_step_returns(reward_sequence, self._reward_decay)
        log_action_prob_old = torch.from_numpy(log_action_prob_sequence).to(self._device)
        for _ in range(self._num_training_iterations):
            action_prob = self._policy_model(states).gather(1, actions.unsqueeze(1)).squeeze()  # (N, 1)
            ratio = torch.exp(torch.log(action_prob) - log_action_prob_old)
            clipped_ratio = torch.clamp(ratio, 1-self._clip_ratio, 1+self._clip_ratio)
            loss = -(torch.min(ratio*returns, clipped_ratio*returns)).mean()
            self._policy_optimizer.zero_grad()
            loss.backward()
            self._policy_optimizer.step()
