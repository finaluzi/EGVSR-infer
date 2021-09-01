from torch.utils.data import DataLoader
from .lr_folder_dataset import LRFolderDataset


# def create_dataloader(opt, dataset_idx='train'):
#     # setup params
#     data_opt = opt['dataset'].get(dataset_idx)

#     # -------------- loader for testing -------------- #
#     if dataset_idx.startswith('test'):
#         # create data loader
#         dataset = LRFolderDataset(
#             data_opt, scale=opt['scale'])
#         loader = DataLoader(
#             dataset=dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=data_opt['num_workers'],
#             pin_memory=data_opt['pin_memory'])

#     else:
#         raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

#     return loader


def create_data_set(opt, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)

    # -------------- loader for testing -------------- #
    if dataset_idx.startswith('test'):
        # create data loader
        dataset = LRFolderDataset(
            data_opt, opt, scale=opt['scale'])

    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return dataset


def create_dataloader_from_set(opt, dataset, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)
    # -------------- loader for testing -------------- #
    if dataset_idx.startswith('test'):
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])
    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return loader
