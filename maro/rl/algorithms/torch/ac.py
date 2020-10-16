# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from maro.rl.algorithms.torch.policy_optimization import PolicyOptimizationWithValueModel, \
    PolicyOptimizationWithCombinedModel
from maro.rl.utils.trajectory_utils import get_lambda_returns


class ActorCritic(PolicyOptimizationWithValueModel):
    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)  # (N, state_dim)
        state_values = self._value_model(states)
        state_values_numpy = state_values.numpy()
        return_est = get_lambda_returns(reward_sequence, self._reward_decay, self._lamb, k=self._k,
                                        values=state_values_numpy)
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = self._policy_model(states).gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
        policy_loss = -(torch.log(action_prob) * (return_est - state_values)).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # value model training
        value_loss = self._value_loss_func(state_values, return_est)
        self._value_optimizer.zero_grad()
        value_loss.backward()
        self._value_optimizer.step()


class ActorCriticWithSharedLayers(PolicyOptimizationWithCombinedModel):
    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)  # (N, state_dim)
        state_values, action_distribution = self._policy_value_model(states)
        state_values_numpy = state_values.numpy()
        return_est = get_lambda_returns(reward_sequence, self._reward_decay, self._lamb, k=self._k,
                                        values=state_values_numpy)
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = action_distribution.gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
        policy_loss = -(torch.log(action_prob) * (return_est - state_values)).mean()
        value_loss = self._value_loss_func(state_values, return_est)
        loss = policy_loss + value_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
