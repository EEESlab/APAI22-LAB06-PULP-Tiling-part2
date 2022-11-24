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

def print_test_vector(x):
    x = x.numpy()
    try:
        np.set_printoptions(threshold=sys.maxsize,formatter={'int': lambda x: hex(np.uint8(x)) if (x < 0) else hex(np.uint8(x)), } )
    except TypeError:
        np.set_printoptions(threshold=sys.maxsize)
    s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(
        ",\n      dtype=int32)", "").replace(", dtype=torch.int32)", "").replace(", dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=uint32)", "").replace("tensor(", "")
    return s

def create_h_file(vector, name):
    tk = OrderedDict([])
    tk['vector'] = print_test_vector(vector)
    tk['dimension'] = vector.flatten().numpy().shape[0]
    tk['vector_name'] = name
    root = os.path.dirname(__file__)
    tmpl = Template(filename=os.path.join(root, "vector_template.h"))
    s_h = tmpl.render(**tk)
    save_string = os.path.join(root, 'Inc/'+ name +'.h') 
    print("creating this file:", save_string)
    with open(save_string, "w") as f:
        f.write(s_h)

def borders(bits, signed = False):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1 if signed else 2 ** bits - 1
    return low, high

def clip(x, bits, signed=False):
    low, high = borders(bits, signed)
    x[x > high] = high
    x[x < low] = low
    return x

def create_input(channels, spatial_dim):
    low, high = borders(8)
    size = (1, channels, spatial_dim, spatial_dim)
    dt = torch.int32
    return torch.randint(low=0, high=100, size=size).to(dtype=dt)

def create_weight(channels):
    low, high = borders(8)
    size = (channels, channels , 1, 1)
    dt = torch.int32
    return torch.randint(low=0, high=5, size=size).to(dtype=dt)

def create_bias(channels):
    size = (channels,1)
    dt = torch.int32
    return torch.randint(low=0, high=1, size=size).flatten().to(dtype=dt)

def create_layer( channels, spatial_dim, input=None, weight=None, batchnorm_params=None):
    x = input if input is not None else create_input(channels, spatial_dim)
    x_save = x.permute(0, 2, 3, 1).flatten()
    create_h_file(x_save, "input")

    w = weight if weight is not None else create_weight(channels)
    create_h_file(w, "weights")
    b = create_bias(channels)
    
    y = F.conv2d(input=x, weight=w, bias=b, stride=1, padding=0, groups=1)
    y = y >> 8
    y = clip(y, 8)
    y_save = y.permute(0, 2, 3, 1) 
    y_save = y_save.flatten()
    create_h_file(y_save, "output")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-shape', '-ks', dest='kernel_shape', type=int, choices=[1], default=1, help='Shape of the kernel. Choices is only 1 for single buffer implementation')
    parser.add_argument('--channels', '-c', type=int, default=1, help='Number of input and output channels')
    parser.add_argument('--output-spatial-dimensions', '-osd', dest='spatial_dimensions', type=int, default=1, help='Output spatial dimension')
    args = parser.parse_args()
    create_layer(channels = args.channels, spatial_dim = args.spatial_dimensions)
