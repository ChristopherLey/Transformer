Transformer from Scratch
========================
by Christopher Ley

How to use
----------
To replicate the environment I used you can install it from the environment.yml file:
~~~console
conda env create -f ./environment.yml
~~~
This is not necessary if you already have an environment with the required dependencies
(The main ones being: _pytorch_, _pytorch-lightning_, _pyyaml_)

Once you have a valid environment, you can simply run:
~~~console
python train.py
~~~
to train the model. If you want to run the model with a different configuration simply specify:
~~~console
python train.py --config ./config.yaml
~~~
A config file specified by the `--config` field should be a path to a valid `.yaml` file using the YAML syntax rules,
a valid config file must contain the following fields, values are completely configurable:
~~~yaml
batch_size: 256
epochs: 2
lr: 3e-4
data_path:  './data/input.txt'
block_size: 256
num_heads: 8
embedding_dim: 512
num_blocks: 6
num_workers: 12
~~~

Background
----------
This is a reproduction of the paper [_Attention is all you need_](https://arxiv.org/abs/1706.03762) with some minor
tweaks (like learnable positional encoding). This is mainly a development to improve my understanding of the
architecture and how it relates to [_Graph Attention Networks_](https://arxiv.org/abs/1710.10903) (GAT) as highlighted in
[_Everything is Connected: Graph Neural Networks_](https://arxiv.org/abs/2301.08210).

The main point is the Transformer
operates over a fully connected graph that learns the importance of its connections via dot product attention. Using an
attention mask is essentially providing an adjacency matrix to this graph. Dot product attention is highly
parallelisable so exploiting this knowledge is useful for creating scalable GAT networks.

Transformer Notes
=================
Model Architecture
------------------
<img src="./static/model_architecture.png" height="700" title="Model Architecture" alt="A diagram of one block of the transformer">

Multihead Attention
-------------------

<img src="./static/multihead_attention.png" height="400" title="Model Architecture" alt="A diagram of one block of the transformer">

Scaled Dot Product Attention
----------------------------

<img src="./static/scaled_dot_product_attention.png" height="400" title="Model Architecture" alt="A diagram of one block of the transformer">
