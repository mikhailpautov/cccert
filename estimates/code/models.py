# +
import torch
import torch.nn as nn
import math
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import *
# -

from .fmt import get_mag

# +
class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        return (input - means) / sds



def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]
# -

ARCHITECTURES = ["resnet18", "resnet50", "cifar_resnet20", "cifar_resnet110", "mnist_43"]


class ResNet18Cifar10(pl.LightningModule):
    def __init__(self, num_classes, model_type, noise_sd, lr):
        super().__init__()

        self.num_classes = num_classes
        self.type = model_type
        self.fmt = 'fmt' in model_type
        self.noise_sd = noise_sd
        self.lr = lr
        self.resnet = torchvision.models.resnet18(pretrained=False,
                num_classes=num_classes)
        inch = 3
        if self.fmt:
            inch = 6
        self.resnet.conv1 = torch.nn.Conv2d(inch, 64, kernel_size=(3,3),
                stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.fmt:
            # add magnitude channel
            magnitude = get_mag(x)

        x = x + torch.randn_like(x, device='cuda') * self.noise_sd
        if self.fmt:
            # concat mag after adding noise
            x = torch.cat((x, magnitude), dim=1)

        y_hat = self.resnet(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, logger=True)

        pred = y_hat.data.max(1)[1]
        acc = torch.sum(pred == y) / x.shape[0]
        # self.log('train_batch_acc', acc, on_step=True, on_epoch=True,
        #         prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.fmt:
            # add magnitude channel
            magnitude = get_mag(x)

        # x = x + torch.randn_like(x, device='cuda') * self.noise_sd
        if self.fmt:
            # concat mag after adding noise
            x = torch.cat((x, magnitude), dim=1)

        y_hat = self.resnet(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('val_batch_loss', loss)
        pred = y_hat.data.max(1)[1]
        acc = torch.sum(pred == y) / x.shape[0]
        return (loss, acc)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_acc = 0.
            total_loss = 0.

            for (loss, acc) in outputs:
                total_num += 1
                total_acc += acc
                total_loss += loss

            acc = total_acc / total_num
            loss = total_loss / total_num
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)

            if self.current_epoch % 10 == 0:
                torch.save(self.state_dict(),
                        f'{self.trainer.log_dir}/checkpoints/epoch{self.current_epoch}.ckpt')


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet.parameters(), lr=self.lr,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                self.trainer.max_epochs)
        return [optim], [scheduler]


# +
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def Conv4FC3():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


# +

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


# -

def get_architecture(arch_type: str, num_classes: int):
    if arch_type == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3,3),
                stride=1, padding=1, bias=False)
    
    elif arch_type == "resnet50":
        raise NotImplementedError('resnet50 for imagenet only, do not have it yet')
        
    elif arch_type == "cifar_resnet20":
        model = resnet(depth=20, num_classes=num_classes)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        
    elif arch_type == "cifar_resnet110":
        model = resnet(depth=110, num_classes=num_classes)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
    elif arch_type == "mnist_43":
        model = Conv4FC3()
        normalize_layer = get_normalize_layer("mnist")
        model = torch.nn.Sequential(normalize_layer, model)
        
        
    else:
        model = None
    return model 


