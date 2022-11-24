# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
# 
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import importlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
from copy import deepcopy
from mako.template import Template
from collections import OrderedDict

def create_layer( channels, spatial_dim):
    MACs = spatial_dim*spatial_dim*channels*channels
    mem = spatial_dim*spatial_dim*channels*2+channels*channels
    print("Mem: {} MACs: {}".format(mem, MACs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('channels', type=int, help='Number of input and output channels')
    parser.add_argument('spatial_dimensions', type=int, help='Spatial dimension')
    args = parser.parse_args()
    create_layer(channels = args.channels, spatial_dim = args.spatial_dimensions)
