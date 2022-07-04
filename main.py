import argparse
import copy
import datetime
import json
import logging
import os
import random

import numpy as np
import torch
import torch.optim as optim

from model import ModelFedX
from utils import F, get_dataloader, mkdirs, partition_data, test_linear_fedX


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet18", help="neural network used in training"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="dataset used for training"
    )
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument(
        "--partition", type=str, default="noniid", help="the data partitioning strategy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.1)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of local epochs")
    parser.add_argument(
        "--n_parties",
        type=int,
        default=10,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "--comm_round",
        type=int,
        default=100,
        help="number of maximum communication roun",
    )
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--datadir", type=str, required=False, default="./data/", help="Data directory"
    )
    parser.add_argument(
        "--reg", type=float, default=1e-5, help="L2 regularization strength"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    parser.add_argument(
        "--log_file_name", type=str, default=None, help="The log file name"
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument(
        "--out_dim",
        type=int,
        default=256,
        help="the output dimension for the projection layer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="the temperature parameter for contrastive loss",
    )
    parser.add_argument(
        "--tt",
        type=float,
        default=0.1,
        help="the temperature parameter for js loss in teacher model",
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=0.1,
        help="the temperature parameter for js loss in student model",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="how many clients are sampled in each round",
    )
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device="cpu"):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = ModelFedX(args.model, args.out_dim, net_configs)
        net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def nt_xent(x1, x2, t=0.1):
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (
        torch.ones_like(sim_matrix)
        - torch.eye(2 * batch_size, device=sim_matrix.device)
    ).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def js_loss(x1, x2, xa, t=0.1, t2=0.01):
    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (
        F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)
    ) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0


def train_net_fedx(
    net_id,
    net,
    global_net,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs,
    lr,
    args_optimizer,
    temperature,
    args,
    round,
    device="cpu",
):
    net.cuda()
    global_net.cuda()
    logger.info("Training network %s" % str(net_id))
    logger.info("n_training: %d" % len(train_dataloader))
    logger.info("n_test: %d" % len(test_dataloader))
    if args_optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=args.reg,
        )
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=args.reg,
            amsgrad=True,
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    net.train()
    global_net.eval()
    random_loader = copy.deepcopy(train_dataloader)
    random_dataloader = iter(random_loader)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x1, x2, target, _) in enumerate(train_dataloader):
            x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
            optimizer.zero_grad()
            target = target.long()

            try:
                random_x, _, _, _ = random_dataloader.next()
            except:
                random_dataloader = iter(random_loader)
                random_x, _, _, _ = random_dataloader.next()
            random_x = random_x.cuda()

            all_x = torch.cat((x1, x2, random_x), dim=0).cuda()
            _, proj1, pred1 = net(all_x)
            with torch.no_grad():
                _, proj2, pred2 = global_net(all_x)

            pred1_original, pred1_pos, pred1_random = pred1.split(
                [x1.size(0), x2.size(0), random_x.size(0)], dim=0
            )
            proj1_original, proj1_pos, proj1_random = proj1.split(
                [x1.size(0), x2.size(0), random_x.size(0)], dim=0
            )
            proj2_original, proj2_pos, proj2_random = proj2.split(
                [x1.size(0), x2.size(0), random_x.size(0)], dim=0
            )

            nt_local = nt_xent(proj1_original, proj1_pos, args.temperature)
            nt_global = nt_xent(pred1_original, proj2_pos, args.temperature)
            loss_nt = nt_local + nt_global

            js_global = js_loss(
                pred1_original, pred1_pos, proj2_random, args.temperature, args.tt
            )
            js_local = js_loss(
                proj1_original, proj1_pos, proj1_random, args.temperature, args.ts
            )
            loss_js = js_global + js_local

            loss = loss_nt + loss_js
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info("Epoch: %d Loss: %f" % (epoch, epoch_loss))
    net.eval()
    logger.info(" ** Training complete **")


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


args = get_args()
mkdirs(args.logdir)
mkdirs(args.modeldir)
if args.log_file_name is None:
    argument_path = f"{args.dataset}-{args.batch_size}-{args.n_parties}-{args.temperature}-{args.tt}-{args.ts}-{args.epochs}_arguments-%s.json" % datetime.datetime.now().strftime(
        "%Y-%m-%d-%H%M-%S"
    )
else:
    argument_path = args.log_file_name + ".json"
with open(os.path.join(args.logdir, argument_path), "w") as f:
    json.dump(str(args), f)
device = torch.device(args.device)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if args.log_file_name is None:
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
logger.info(device)

seed = args.init_seed
logger.info("#" * 100)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

logger.info("Partitioning data")
(
    X_train,
    y_train,
    X_test,
    y_test,
    net_dataidx_map,
    traindata_cls_counts,
) = partition_data(
    args.dataset,
    args.datadir,
    args.logdir,
    args.partition,
    args.n_parties,
    beta=args.beta,
)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []

if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

n_classes = len(np.unique(y_train))

(
    train_dl_global,
    val_dl_global,
    test_dl,
    train_ds_global,
    _,
    test_ds_global,
) = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size * 2)


def local_train_net(
    nets,
    args,
    net_dataidx_map,
    train_dl_local_dict,
    val_dl_local_dict,
    train_dl=None,
    test_dl=None,
    global_model=None,
    prev_model_pool=None,
    round=None,
    device="cpu",
):
    pass

    if global_model:
        global_model.cuda()

    n_epoch = args.epochs
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        logger.info(
            "Training network %s. n_training: %d" % (str(net_id), len(dataidxs))
        )
        train_dl_local = train_dl_local_dict[net_id]
        val_dl_local = val_dl_local_dict[net_id]
        train_net_fedx(
            net_id,
            net,
            global_model,
            train_dl_local,
            val_dl_local,
            test_dl,
            n_epoch,
            args.lr,
            args.optimizer,
            args.temperature,
            args,
            round,
            device=device,
        )

    if global_model:
        global_model.to("cpu")

    return nets


print("len train_dl_global:", len(train_ds_global))
train_dl = None
data_size = len(test_ds_global)

logger.info("Initializing nets")
nets, local_model_meta_data, layer_type = init_nets(
    args.net_config, args.n_parties, args, device="cpu"
)
global_models, global_model_meta_data, global_layer_type = init_nets(
    args.net_config, 1, args, device="cpu"
)

global_model = global_models[0]
n_comm_rounds = args.comm_round

train_dl_local_dict = {}
val_dl_local_dict = {}
net_id = 0

for net in nets:
    dataidxs = net_dataidx_map[net_id]
    train_dl_local, val_dl_local, _, _, _, _ = get_dataloader(
        args.dataset, args.datadir, args.batch_size, args.batch_size * 2, dataidxs
    )
    train_dl_local_dict[net_id] = train_dl_local
    val_dl_local_dict[net_id] = val_dl_local
    net_id += 1


for round in range(n_comm_rounds):
    logger.info("in comm round:" + str(round))
    party_list_this_round = party_list_rounds[round]

    global_w = global_model.state_dict()
    nets_this_round = {k: nets[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.load_state_dict(global_w)

    local_train_net(
        nets_this_round,
        args,
        net_dataidx_map,
        train_dl_local_dict,
        val_dl_local_dict,
        train_dl=train_dl,
        test_dl=test_dl,
        global_model=global_model,
        device=device,
    )

    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
    fed_avg_freqs = [
        len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)
    ]

    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.load_state_dict(copy.deepcopy(global_w))
    global_model.cuda()
    test_acc_1, test_acc_5 = test_linear_fedX(global_model, val_dl_global, test_dl)
    logger.info(">> Global Model Test accuracy Top1: %f" % test_acc_1)
    logger.info(">> Global Model Test accuracy Top5: %f" % test_acc_5)

torch.save(
    global_model.state_dict(),
    args.modeldir + "globalmodel" + args.log_file_name + ".pth",
)
torch.save(
    nets[0].state_dict(), args.modeldir + "localmodel0" + args.log_file_name + ".pth"
)
