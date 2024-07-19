import numpy as np
# import torch
def check_dict_old(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                print('{:<60} {:<20}'.format(key, str(value)))
            else:
                print('{:<60} {:<20}'.format(key, str(value.shape)))
        # elif isinstance(value, torch.Tensor):
        #     print('{:<60} {:<20}'.format(key, str(value.shape)))
        elif isinstance(value, dict):
            print('{:<60} {:<20}'.format(key, str(value.keys())))
        elif isinstance(value, list):
            print('{:<60} {:<20}'.format(key, 'list: (' + str(len(value))+')'))
        else:
            print('{:<60} {:<20}'.format(key, str(value)))

def check_dict(d, key_width=40, value_width=20):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                print(f'{key:<{key_width}} {str(value):<{value_width}}')
            else:
                print(f'{key:<{key_width}} {str(value.shape):<{value_width}}')
        # elif isinstance(value, torch.Tensor):
        #     print(f'{key:<{key_width}} {str(value.shape):<{value_width}}')
        elif isinstance(value, dict):
            print(f'{key:<{key_width}} {str(value.keys()):<{value_width}}')
        elif isinstance(value, list):
            if len(value) < 10:
                print(f'{key:<{key_width}} {value}')
            else:
                print(f'{key:<{key_width}} list: ({str(len(value))})')

        else:
            print(f'{key:<{key_width}} {str(value):<{value_width}}')
