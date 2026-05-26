"""DRL-VO policy: CustomCNN feature extractor and inference wrapper.

Adapted from TempleRAIL/drl_vo_nav (humble branch).
See LICENSE for upstream terms.
"""

from __future__ import annotations

import sys

import gym
import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.modules.setdefault("custom_cnn_full", sys.modules[__name__])


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CustomCNN(BaseFeaturesExtractor):
    """ResNet-based feature extractor for DRL-VO observations."""

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 256
    ) -> None:
        super().__init__(observation_space, features_dim)

        block = _Bottleneck
        layers = [2, 1, 1]
        norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1, stride=4),
            nn.BatchNorm2d(512),
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * block.expansion + 2, features_dim),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        for m in self.modules():
            if isinstance(m, _Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layer_list = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_list.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layer_list)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ped_pos = observations[:, :12800]
        scan = observations[:, 12800:19200]
        goal = observations[:, 19200:]

        ped_in = ped_pos.reshape(-1, 2, 80, 80)
        scan_in = scan.reshape(-1, 1, 80, 80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)

        x = self.relu(self.bn1(self.conv1(fusion_in)))
        x = self.maxpool(x)

        identity3 = self.downsample3(x)
        x = self.layer1(x)
        identity2 = self.downsample2(x)
        x = self.layer2(x)
        x = self.relu2(self.conv2_2(x) + identity2)
        x = self.layer3(x)
        x = self.relu3(self.conv3_2(x) + identity3)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        goal_out = goal.reshape(-1, 2)
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        return self.linear_fc(fc_in)


_POLICY_KWARGS = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


class DrlVoPolicy:
    """Inference wrapper around the DRL-VO PPO model."""

    def __init__(
        self,
        weights_path: str,
        range_limit: float = 30.0,
        vx_limit: float = 0.5,
        wz_limit: float = 0.7,
    ) -> None:
        self.range_limit = range_limit
        self.vx_limit = vx_limit
        self.wz_limit = wz_limit
        self.model = PPO.load(weights_path, device="cpu")

    def compute_velocity_commands(
        self,
        scan_history: list[float],
        current_scan: list[float],
        sub_goal: list[float],
    ) -> tuple[float, float]:
        """Return (vx, wz) from scan history, current scan, and sub-goal."""
        len_scan = len(current_scan)
        scan_arr = np.array(
            current_scan[
                int(len_scan / 2 - len_scan / 9) : int(len_scan / 2 + len_scan / 9)
            ]
        )
        scan_nonzero = scan_arr[scan_arr != 0]
        min_scan_dist = float(np.amin(scan_nonzero)) if scan_nonzero.size != 0 else 30.0

        if min_scan_dist <= 0.45:
            return 0.0, self.wz_limit

        ped_map = np.zeros(12800, dtype=np.float32)

        temp = np.array(scan_history, dtype=np.float32)
        scan_avg = np.zeros((20, 80), dtype=np.float32)
        for n in range(10):
            scan_tmp = temp[n * len_scan : (n + 1) * len_scan]
            if len(scan_tmp) > 0:
                step = int(len_scan / 80)
                for i in range(80):
                    scan_avg[2 * n, i] = np.min(scan_tmp[i * step : (i + 1) * step])
                    scan_avg[2 * n + 1, i] = np.mean(
                        scan_tmp[i * step : (i + 1) * step]
                    )

        scan_avg = scan_avg.reshape(1600)
        scan_map = np.matlib.repmat(scan_avg, 1, 4).reshape(6400)
        scan_map = 2.0 * (scan_map - 0.0) / (self.range_limit - 0.0) + (-1.0)

        goal_arr = np.array(sub_goal, dtype=np.float32)
        goal_scaled = 2.0 * (goal_arr - (-2.0)) / (2.0 - (-2.0)) + (-1.0)

        observation = np.concatenate((ped_map, scan_map, goal_scaled), axis=None)
        action, _ = self.model.predict(observation, deterministic=True)

        if self.vx_limit >= 0.75:
            vx_max = self.vx_limit if min_scan_dist >= 2.5 else self.vx_limit / 1.5
        else:
            vx_max = self.vx_limit
        vx_min = 0.0

        if self.wz_limit >= 1.0:
            vz_min = -self.wz_limit / 1.5
            vz_max = self.wz_limit / 1.5
        else:
            vz_min = -self.wz_limit
            vz_max = self.wz_limit

        vx = float((action[0] + 1.0) * (vx_max - vx_min) / 2.0 + vx_min)
        wz = float((action[1] + 1.0) * (vz_max - vz_min) / 2.0 + vz_min)
        return vx, wz
