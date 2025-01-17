import  torch, os
import  numpy as np
from    FSS import FSSDataset
import  scipy.stats
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from learner import Learner
from meta import Meta

from oblog import log_args, log_train_acc, log_test_acc

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    log_args(args)

    config = [
        ('conv2d', [16, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('conv2d', [32, 16, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [32, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [16, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('conv2d', [2, 16, 3, 3, 1, 1]),
        ('relu', [True]),
        # ('max_pool2d', [2, 2, 0]),
        # ('conv2d', [32, 32, 3, 3, 1, 0]),
        # ('relu', [True]),
        # ('bn', [32]),
        # ('max_pool2d', [2, 2, 0]),
        # ('conv2d', [32, 32, 3, 3, 1, 0]),
        # ('relu', [True]),
        # ('bn', [32]),
        # ('max_pool2d', [2, 2, 0]),
        # ('conv2d', [32, 32, 3, 3, 1, 0]),
        # ('relu', [True]),
        # ('bn', [32]),
        # ('max_pool2d', [2, 1, 0]),
        # ('flatten', []),
        # ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    learner = Learner(config, args.imgc, args.imgsz).to(device)
    loss_fn = F.cross_entropy
    maml = Meta(args, learner, loss_fn).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = FSSDataset(args.data, mode='train', k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = FSSDataset(args.data, mode='test', k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                log_train_acc(accs)
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                log_test_acc(accs)
                print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n-way', type=int, help='n way', default=5)
    argparser.add_argument('--k-spt', type=int, help='k shot for support set', default=2)
    argparser.add_argument('--k-qry', type=int, help='k shot for query set', default=4)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task-num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta-lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update-lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update-step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update-step-test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--data', type=str, help='Dataset path', required=True)
    argparser.add_argument('--num-workers', type=int, help="Number of worker threads", default = 1)

    args = argparser.parse_args()

    main()
