from torch import nn
import torch
from torch import optim
from .base_model import BaseModel
from hw2.roble.infrastructure.utils import *
from hw1.roble.infrastructure import pytorch_util as ptu


class FFModel(nn.Module, BaseModel):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **params):
        super(FFModel, self).__init__()

        self._delta_network = ptu.build_mlp(
            input_size=self._ob_dim + self._ac_dim,
            output_size=self._ob_dim,
            params=self._network
        )
        self._delta_network.to(ptu.device)
        self._optimizer = optim.Adam(
            self._delta_network.parameters(),
            self._learning_rate,
        )
        self._loss = nn.MSELoss()
        self._obs_mean = None
        self._obs_std = None
        self._acs_mean = None
        self._acs_std = None
        self._delta_mean = None
        self._delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self._obs_mean = ptu.from_numpy(obs_mean)
        self._obs_std = ptu.from_numpy(obs_std)
        self._acs_mean = ptu.from_numpy(acs_mean)
        self._acs_std = ptu.from_numpy(acs_std)
        self._delta_mean = ptu.from_numpy(delta_mean)
        self._delta_std = ptu.from_numpy(delta_std)

    def forward(
                self,
                obs_unnormalized,
                acs_unnormalized,
                obs_mean,
                obs_std,
                acs_mean,
                acs_std,
                delta_mean,
                delta_std,
        ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        # obs_unnormalized = torch.tensor(obs_unnormalized, dtype=torch.float32)
        # acs_unnormalized = torch.tensor(acs_unnormalized, dtype=torch.float32)
        # obs_mean = torch.tensor(obs_mean, dtype=torch.float32)
        # obs_std = torch.tensor(obs_std, dtype=torch.float32)
        # acs_mean = torch.tensor(acs_mean, dtype=torch.float32)
        # acs_std = torch.tensor(acs_std, dtype=torch.float32)
        # delta_mean = torch.tensor(delta_mean, dtype=torch.float32)
        # delta_std = torch.tensor(delta_std, dtype=torch.float32)
        # normalize input data to mean 0, std 1
        # obs_normalized = # TODO(Q1)
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        # obs_normalized = (obs_unnormalized - obs_mean) / (obs_std + 1e-8)
        # acs_normalized = # TODO(Q1)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)
        # acs_normalized = (acs_unnormalized - acs_mean) / (acs_std + 1e-8)
        
        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        # delta_pred_normalized = # TODO(Q1)
        delta_pred_normalized = self._delta_network(concatenated_input)
        # next_obs_pred = # TODO(Q1)
        delta_pred_unnormalized = unnormalize(delta_pred_normalized, 
                                              delta_mean, 
                                              delta_std)
        next_obs_pred = obs_unnormalized + delta_pred_unnormalized
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        # prediction = # TODO(Q1) get numpy array of the predicted next-states (s_t+1)
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        if len(obs.shape) == 1 or len(acs.shape) == 1:
            obs = np.squeeze(obs)[None]
            acs = np.squeeze(acs)[None]
            
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        new_data_statistics = {
            k: ptu.from_numpy(v) for k, v in data_statistics.items()
        }
        prediction, _ = self(obs, acs, **new_data_statistics)
        return ptu.to_numpy(prediction)

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """
        # target = # TODO(Q1) compute the normalized target for the model.
        # # Hint: you should use `data_statistics['delta_mean']` and
        # # `data_statistics['delta_std']`, which keep track of the mean
        # # and standard deviation of the model.
        
        # delta_unnormalized = (next_observations - observations)
        # delta_unnormalized_tensor = torch.tensor(delta_unnormalized, dtype=torch.float32)
        # delta_mean_tensor = torch.tensor(data_statistics['delta_mean'], dtype=torch.float32)
        # delta_std_tensor = torch.tensor(data_statistics['delta_std'], dtype=torch.float32)
        # target = (delta_unnormalized_tensor - delta_mean_tensor) / (delta_std_tensor + 1e-8)  # Adding a small epsilon to avoid division by zero
        
        # loss = # TODO(Q1) compute the loss
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        target = normalize(
        next_observations - observations,
        data_statistics['delta_mean'],
        data_statistics['delta_std'],
        )

        obs = ptu.from_numpy(observations)
        acs = ptu.from_numpy(actions)
        torch_data_statistics = {
            k: ptu.from_numpy(v) for k, v in data_statistics.items()
        }
        _, delta_pred_normalized = self(obs, acs, **torch_data_statistics)

        target = ptu.from_numpy(target)
        loss = self._loss(delta_pred_normalized, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # self.update_statistics(obs_mean=observations.mean(axis=0),
        #                        obs_std=observations.std(axis=0),
        #                        acs_mean=actions.mean(axis=0),
        #                        acs_std=actions.mean(axis=0),
        #                        delta_mean=delta_unnormalized.mean(axis=0),
        #                        delta_std=delta_unnormalized.std(axis=0))

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
