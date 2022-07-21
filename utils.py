import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets




class Imagenet_idx(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(Imagenet_idx, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            pos = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target, index



class Imagenet_idx_pair(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(Imagenet_idx_pair, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            pos1 = self.transform(image)
            pos2 = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos1, pos2, target, index



class Imagenet_idx_pair_transformone(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform_simple=None, transform_hard=None, target_transform=None):
        super(Imagenet_idx_pair_transformone, self).__init__(root, transform_simple, target_transform)
        self.transform_hard = transform_hard

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            pos1 = self.transform(image)
            pos2 = self.transform(image)
        if self.transform_hard is not None:
            pos1_hard = self.transform_hard(image)
            pos2_hard = self.transform_hard(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos1, pos2, pos1_hard, pos2_hard, target, index




class ResNet_ft(nn.Module):
    def __init__(self, model, fc, args=None):
        super(ResNet_ft, self).__init__()
        self.model = model
        self.fc = fc
        self.args = args

        if 'mlp' in args and args.mlp: # use mlp
            hidden_layer = 2048
            self.mlp = nn.Sequential(nn.Linear(hidden_layer, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))


    def forward(self, image, return_feature=False):
        feature = self.model(image)

        if return_feature:
            return feature

        output = self.fc(feature)
        return output




class ResNet_ft_eqinv(nn.Module):
    def __init__(self, model, fc, mask_layer=None, args=None):
        super(ResNet_ft_eqinv, self).__init__()
        self.model = model
        self.fc = fc
        if args.opt_mask:
            self.mask_layer = torch.nn.Parameter(mask_layer)
        else:
            self.mask_layer = mask_layer
        self.args = args
        if mask_layer is not None: # use mask
            self.activation_map = activation_map(args.activat_type)
            self.scaler = 10

        if 'mlp' in args and args.mlp: # use mlp
            hidden_layer = 2048
            self.mlp = nn.Sequential(nn.Linear(hidden_layer, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))



    def forward(self, image, return_feature=False, return_masked_feature=False):
        feature = self.model(image)

        if return_feature:
            return feature

        if self.mask_layer is not None:
            masked_feature_erm = F.normalize(self.activation_map.apply(self.mask_layer)*feature, dim=-1) * self.scaler # nomalize for numeral stability
            masked_feature_inv = F.normalize(self.activation_map.apply(self.mask_layer)*feature.detach(), dim=-1) * self.scaler # invariant loss not backward to backbone, thus detach the feature
        else:
            raise NotImplementedError

        output = self.fc(masked_feature_erm)

        if return_masked_feature:
            if self.mask_layer is None:
                raise NotImplementedError
            else:
                if self.args.mlp:
                    return self.mlp(masked_feature_erm), masked_feature_inv, output
                else:
                    return masked_feature_inv, output
        else:
            return output





class activation_map():
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def apply(self, x, soft=False):
        if self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_type == 'ident':
            return x
        elif self.activation_type == 'gumbel':
            if soft:
                x_hard = F.gumbel_softmax(x, tau=1, hard=False)
            else:
                x_hard = F.gumbel_softmax(x, tau=1, hard=True)
            return x_hard[:,1].squeeze().unsqueeze(0)


from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def write_log(print_str, log_file, print_=False):
    if print_:
        print(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


import random
def set_seed(seed):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


