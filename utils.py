""" 
Utility functions to load, save, log, and process data.

Some of the codes in this file are excerpted from the original work
https://github.com/QinbinLi/MOON/blob/main/utils.py

"""

import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from datasets import CIFAR10_truncated, SVHN_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception:
        pass


def set_logger(args):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    args.log_file_name = (
        f"{args.dataset}-{args.batch_size}-{args.n_parties}-{args.temperature}-{args.tt}-{args.ts}-{args.epochs}_log-%s"
        % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    )
    log_path = args.log_file_name + ".log"
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.DEBUG,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_truncated(datadir, split="train", download=True, transform=transform)
    svhn_test_ds = SVHN_truncated(datadir, split="test", download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print("mean:", np.mean(data_list))
    print("std:", np.std(data_list))
    logger.info("Data statistics: %s" % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    """Data partitioning to each local party according to the beta distribution"""
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == "svhn":
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)

    n_train = y_train.shape[0]

    # Paritioning option
    if partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


class Net(nn.Module):
    """Prediction head class for linear evaluation"""

    def __init__(self, dim_input, num_class):
        super(Net, self).__init__()
        self.fc = nn.Linear(dim_input, num_class, bias=True)

    def forward(self, x):
        out = self.fc(x)
        return out


def test_linear_fedX(net, memory_data_loader, test_data_loader):
    """Linear evaluation code for FedX"""
    net.eval()
    feature_bank = []

    # Save training data's embeddings into the feature_bank.
    with torch.no_grad():
        for data, _, target, _ in memory_data_loader:
            feature, _, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous().cuda()
        feature_labels = torch.tensor(memory_data_loader.dataset.target, device=feature_bank.device)

    linear_ds = TensorDataset(feature_bank, feature_labels)
    linear_loader = DataLoader(linear_ds, batch_size=64, shuffle=True)

    # Save test data's embeddings into the feature_bank_test
    feature_bank_test = []
    with torch.no_grad():
        for data, _, target, _ in test_data_loader:
            feature_test, _, _ = net(data.cuda(non_blocking=True))
            feature_bank_test.append(feature_test)
        feature_bank_test = torch.cat(feature_bank_test, dim=0).contiguous().cuda()
        feature_labels_test = torch.tensor(test_data_loader.dataset.target, device=feature_bank_test.device)

    linear_ds_test = TensorDataset(feature_bank_test, feature_labels_test)
    linear_loader_test = DataLoader(linear_ds_test, batch_size=64, shuffle=True)

    loss_criterion = nn.CrossEntropyLoss()
    linear_net = Net(feature_bank.shape[-1], feature_labels.max().item() + 1)
    linear_net = linear_net.cuda()
    train_optimizer = optim.Adam(linear_net.parameters(), lr=1e-3, weight_decay=1e-6)

    # Train linear layer (fix the backbone network)
    for epoch in range(1, 101):
        for data, target in linear_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = linear_net(data)
            loss = loss_criterion(out, target)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

    # Evaluation
    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
    with torch.no_grad():
        for data, target in linear_loader_test:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = linear_net(data)

            total_num += data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    return total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset == "cifar10":
        dl_obj = CIFAR10_truncated

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            train=True,
            transform=transform_train,
            download=True,
        )
        val_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            train=True,
            transform=transform_test,
            download=False,
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
        val_dl = data.DataLoader(dataset=val_ds, batch_size=test_bs, shuffle=False)

    elif dataset == "svhn":
        dl_obj = SVHN_truncated
        normalize = transforms.Normalize(
            mean=[0.4376821, 0.4437697, 0.47280442],
            std=[0.19803012, 0.20101562, 0.19703614],
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ]
        )
        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            split="train",
            transform=transform_train,
            download=True,
        )
        val_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            split="train",
            transform=transform_test,
            download=False,
        )
        test_ds = dl_obj(datadir, split="test", transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
        val_dl = data.DataLoader(dataset=val_ds, batch_size=test_bs, shuffle=False)

    return (
        train_dl,
        val_dl,
        test_dl,
        train_ds,
        val_ds,
        test_ds,
    )
