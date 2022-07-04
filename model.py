import torch.nn as nn

from resnetcifar import ResNet18_cifar10, ResNet50_cifar10


class ModelFedX(nn.Module):
    def __init__(self, base_model, out_dim, net_configs=None):
        super(ModelFedX, self).__init__()

        if (
            base_model == "resnet50-cifar10"
            or base_model == "resnet50-cifar100"
            or base_model == "resnet50-smallkernel"
            or base_model == "resnet50"
        ):
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            basemodel.fc.in_features
        elif base_model == "resnet18-fmnist":
            basemodel = ResNet18_mnist()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        else:
            raise (
                "Invalid model type. Check the config file and pass one of: resnet18 or resnet50"
            )

        self.projectionMLP = nn.Sequential(
            nn.Linear(self.num_ftrs, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        self.predictionMLP = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )

    def forward(self, x):
        h = self.features(x)

        h.view(-1, self.num_ftrs)
        h = h.squeeze()

        proj = self.projectionMLP(h)
        pred = self.predictionMLP(proj)
        return h, proj, pred
