import os
import os.path as osp
import argparse
from torch.nn.functional import selu
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
                    print('[W] Start Worker output images', self.getName(), end='\r')
                    self.working = True
                    self.hr_dict = val
                    # start = time.perf_counter()
                    self.save_images()
                    # print(
                        # f"save_images Completed Execution in {time.perf_counter() - start} seconds")
                    self.hr_dict = {}
                    self.working = False
                    print('[W] End Worker output images', self.getName(), end='\r')
            time.sleep(0.06)

    def save_images(self):
        hr_seqs = self.hr_dict['hr_seqs']
        start_idx = self.hr_dict['start_idx']
        opt_dict = self.hr_dict['opt']
        opt = opt_dict['opt']
        model_idx = opt_dict['model_idx']
        ds_name = opt_dict['ds_name']
        seq_idx = opt_dict['seq_idx']
        frm_idx = opt_dict['frm_idx'][start_idx:]
        v_max = opt_dict['v_max']
        if opt['test']['save_res']:
            res_dir = osp.join(
                opt['test']['res_dir'], ds_name, model_idx)
            res_seq_dir = osp.join(res_dir, seq_idx)
            data_utils.save_sequence(
                res_seq_dir, hr_seqs, v_max, frm_idx)


class SaveThreadsMgr():
    def __init__(self):
        self.threads = []
        self.save_opt_cache = {}

    def add_thread_start(self):
        t = SaveThread(Queue())
        t.start()
        self.threads.append(t)
        return len(self.threads)

    def save_img_worker_block(self, save_dict):
        # save_dict hr_seqs, start_idx
        has_free_worker = False
        count = 0
        while not has_free_worker:
            print('[W] Finding free worker...', count, end='\r')
            for t in self.threads:
                if not t.working:
                    save_dict['opt'] = self.save_opt_cache
                    t.queue.put(save_dict)
                    has_free_worker = True
                    break
            count += 1
            time.sleep(0.1)

    def join_all(self):
        print('[W] Waiting for Workers...')
        for t in self.threads:
            t.queue.put('end')
            t.join()


def test(opt):
    # logging
    threads = SaveThreadsMgr()
    for _ in range(0, opt['test']['num_save_threads']):
        threads.add_thread_start()

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
        # bind save threads
        model.bind_save_threads(threads, opt['test']['save_images_num'])
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
                # enumerate=5s
                for _, data in enumerate(test_loader):
                    # fetch data
                    lr_data = data['lr'][0]
                    tot_frm, _, _, _ = lr_data.size()
                    if tot_frm-pad_num > pad_num:
                        seq_idx = data['seq_idx'][0]
                        frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                        threads.save_opt_cache = {'opt': opt, 'model_idx': model_idx, 'ds_name': ds_name,
                                                  'seq_idx': seq_idx, 'frm_idx': frm_idx, 'v_max': v_max}
                        model.infer(lr_data)  # thwc|rgb|uint8

                        # threads.save_img_worker_block(save_dict)
                    else:
                        print('[B] Skip batch', test_set.batch_num)

                logger.info('-' * 40)
                test_set.batch_num += 1

    # logging
    threads.join_all()
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
