import os
import os.path as osp
import argparse
import yaml
import time
import torch
from data import create_data_set, create_dataloader_from_set
from models import define_model
from utils import base_utils, data_utils
import threading
import time
from queue import Queue
import warnings
warnings.filterwarnings("ignore")


class SaveThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.hr_dict = {}
        self.working = False
        # self.isDaemon = True

    def run(self):
        while True:
            if not self.queue.empty():
                val = self.queue.get()
                if type(val) is str:
                    print('[W] Worker End', self.getName())
                    break
                else:
                    print('[W] Start Worker output images', self.getName())
                    self.working = True
                    self.hr_dict = val
                    self.save_images()
                    self.hr_dict = {}
                    self.working = False
                    print('[W] End Worker output images', self.getName())
            time.sleep(0.06)

    def save_images(self):
        opt = self.hr_dict['opt']
        model_idx = self.hr_dict['model_idx']
        ds_name = self.hr_dict['ds_name']
        seq_idx = self.hr_dict['seq_idx']
        frm_idx = self.hr_dict['frm_idx']
        hr_seqs = self.hr_dict['hr_seqs']
        v_max = self.hr_dict['v_max']
        if opt['test']['save_res']:
            res_dir = osp.join(
                opt['test']['res_dir'], ds_name, model_idx)
            res_seq_dir = osp.join(res_dir, seq_idx)
            data_utils.save_sequence(
                res_seq_dir, hr_seqs, v_max, frm_idx)


def test(opt):
    # logging
    threads = []
    for _ in range(0, opt['test']['num_save_threads']):
        thread = SaveThread(Queue())
        thread.start()
        threads.append(thread)

    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]

        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)
        pad_num = opt['test']['num_pad_front']
        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            if not dataset_idx.startswith('test'):
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))
            v_max = opt['dataset'][dataset_idx]['max_vertical_res']

            # create data loader
            test_set = create_data_set(opt, dataset_idx=dataset_idx)

            while not test_set.is_end():
                test_loader = create_dataloader_from_set(
                    opt, test_set, dataset_idx=dataset_idx)

                # infer and store results for each sequence
                for _, data in enumerate(test_loader):
                    # fetch data
                    lr_data = data['lr'][0]
                    tot_frm, _, _, _ = lr_data.size()
                    if tot_frm-pad_num > pad_num:
                        seq_idx = data['seq_idx'][0]
                        frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                        # print(lr_data, seq_idx, frm_idx)
                        # infer
                        # print('infer start')
                        hr_seq = model.infer(lr_data)  # thwc|rgb|uint8
                        # print('infer over')

                        save_dict = {'opt': opt, 'model_idx': model_idx, 'ds_name': ds_name,
                                     'seq_idx': seq_idx, 'frm_idx': frm_idx, 'hr_seqs': hr_seq, 'v_max': v_max}
                        # save results (optional)
                        has_free_worker = False
                        while not has_free_worker:
                            for t in threads:
                                if not t.working:
                                    t.queue.put(save_dict)
                                    has_free_worker = True
                                    break
                            time.sleep(0.03)
                    else:
                        print('[B] Skip batch', test_set.batch_num)

                logger.info('-' * 40)
                test_set.batch_num += 1

    # logging
    print('[W] Waiting for Workers...')
    for t in threads:
        t.queue.put('end')
        t.join()
    logger.info('Finish testing')
    logger.info('=' * 40)


if __name__ == '__main__':
    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the option yaml file')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU index, -1 for CPU')
    args = parser.parse_args()

    # ----------------- get options ----------------- #
    print(args.exp_dir)
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = args.exp_dir

    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])

    # logger
    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'

    # ----------------- test ----------------- #
# setup paths
    base_utils.setup_paths(opt, mode='test')

    # run
    opt['is_train'] = False
    test(opt)
