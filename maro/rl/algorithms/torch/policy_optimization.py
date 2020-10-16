# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm


class PolicyOptimization(AbsAlgorithm):
    """Policy optimization algorithm family.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        optimizer_cls: torch optimizer class for the policy model.
        optimizer_params: parameters required for the policy optimizer class.
        num_actions (int): number of possible actions.
        reward_decay (float): reward decay as defined in standard RL terminology.
    """
    def __init__(self,
                 policy_model: nn.Module,
                 optimizer_cls,
                 optimizer_params,
                 num_actions: int,
                 reward_decay: float):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_model = policy_model.to(self._device)
        self._policy_optimizer = optimizer_cls(self._policy_model.parameters(), **optimizer_params)
        self._num_actions = num_actions
        self._reward_decay = reward_decay

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)  # (1, state_dim)
        action_dist = self._policy_model(state).squeeze().numpy()  # (num_actions,)
        action_index = np.random.choice(self._num_actions, p=action_dist)
        return action_index, np.log(action_dist[action_index])

    @abstractmethod
    def train(self, *args, **kwargs):
        return NotImplementedError

    def load_trainable_models(self, policy_model):
        self._policy_model = policy_model

    def dump_trainable_models(self):
        return self._policy_model

    def load_trainable_models_from_file(self, path):
        self._policy_model = torch.load(path)

    def dump_trainable_models_to_file(self, path: str):
        torch.save(self._policy_model.state_dict(), path)


class PolicyOptimizationWithValueModel(PolicyOptimization):
    """Actor Critic algorithm with separate policy and value models (no shared layers).

    The Actor-Critic algorithm base on the policy gradient theorem.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        value_model (nn.Module): model for estimating state values.
        value_loss_func (Callable): loss function for the value model.
        policy_optimizer_cls: torch optimizer class for the policy model.
        policy_optimizer_params: parameters required for the policy optimizer class.
        value_optimizer_cls: torch optimizer class for the value model.
        value_optimizer_params: parameters required for the value optimizer class.
        num_actions (int): number of possible actions.
        reward_decay (float): reward decay as defined in standard RL terminology.
        k (int): number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lamb (float): lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
    """

    def __init__(self,
                 policy_model: nn.Module,
                 value_model: nn.Module,
                 value_loss_func: Callable,
                 policy_optimizer_cls,
                 policy_optimizer_params,
                 value_optimizer_cls,
                 value_optimizer_params,
                 num_actions: int,
                 reward_decay: float,
                 k: int = -1,
                 lamb: float = 1.0):
        super().__init__(policy_model, policy_optimizer_cls, policy_optimizer_params, num_actions, reward_decay)
        self._value_model = value_model.to(self._device)
        self._value_optimizer = value_optimizer_cls(self._value_model.parameters(), **value_optimizer_params)
        self._value_loss_func = value_loss_func
        self._k = k
        self._lamb = lamb

    @abstractmethod
    def train(self, *args, **kwargs):
        return NotImplementedError

    def load_trainable_models(self, model_dict):
        self._policy_model = model_dict["policy"]
        self._value_model = model_dict["value"]

    def dump_trainable_models(self):
        return {"policy": self._policy_model, "value": self._value_model}

    def load_trainable_models_from_file(self, path):
        model_dict = torch.load(path)
        self._policy_model = model_dict["policy"]
        self._value_model = model_dict["value"]

    def dump_trainable_models_to_file(self, path: str):
        torch.save({"policy": self._policy_model.state_dict(), "value": self._value_model.state_dict()}, path)


class PolicyOptimizationWithCombinedModel(AbsAlgorithm):
    def __init__(self,
                 policy_value_model: nn.Module,
                 value_loss_func: Callable,
                 optimizer_cls,
                 optimizer_params,
                 num_actions: int,
                 reward_decay: float,
                 k: int = -1,
                 lamb: float = 1.0):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_value_model = policy_value_model.to(self._device)
        self._optimizer = optimizer_cls(self._policy_value_model.parameters(), **optimizer_params)
        self._value_loss_func = value_loss_func
        self._num_actions = num_actions
        self._reward_decay = reward_decay
        self._k = k
        self._lamb = lamb

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)  # (1, state_dim)
        action_dist = self._policy_value_model(state)[1].squeeze().numpy()  # (num_actions,)
        action_index = np.random.choice(self._num_actions, p=action_dist)
        return action_index, np.log(action_dist[action_index])

    @abstractmethod
    def train(self, *args, **kwargs):
        return NotImplementedError

    def load_trainable_models(self, policy_value_model):
        self._policy_value_model = policy_value_model

    def dump_trainable_models(self):
        return self._policy_value_model

    def load_trainable_models_from_file(self, path):
        self._policy_value_model = torch.load(path)

    def dump_trainable_models_to_file(self, path: str):
        torch.save(self._policy_value_model.state_dict(), path)
