import os
from collections import namedtuple


config = {
    'seed': 1,
    'attr_num': 5,
    'max_code': 200,
    'max_msg': 20,
    'mark_embed': 50,
    'word_embed': 150,
    'hid_dim': 128,
    'attn_num': 64,
    'drop_rate': 0.2,
    'learn_rate': 0.0001,
    'beam_size': 5,
    'batch_size': 128,
    'epochs': 300,
    'patience': 5,
    'model_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'),
    'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'),
}
Config = namedtuple('Config', config)
conf = Config(**config)
