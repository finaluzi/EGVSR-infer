from .vsr_model import VSRModel

# register vsr model
vsr_model_lst = [
    'frvsr',
]


def define_model(opt):
    if opt['model']['name'].lower() in vsr_model_lst:
        model = VSRModel(opt)

    else:
        raise ValueError('Unrecognized model: {}'.format(opt['model']['name']))

    return model
