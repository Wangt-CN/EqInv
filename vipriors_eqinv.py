import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] ='0'
import random
import shutil
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.optim

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import defaultdict


import utils
import utils_cluster
from torchvision.models.resnet import resnet50
from randaugment import RandAugment

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[30, 40], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action="store_true", default=False, help='evaluate?')
parser.add_argument('--name', default=None, type=str,
                    help='exp name')
parser.add_argument('--clip-a', default=None, type=str,
                    help='clip architecture')
parser.add_argument('--adam', action="store_true", default=False, help='use adam optimizer?')
parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
parser.add_argument('--temperature', default=0.1, type=float, help='temperature for contrastive loss')
parser.add_argument('--cont_weight', default=1.0, type=float, help='weight of contrastive loss')
parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')

parser.add_argument('--stage1_model', type=str, default='ipirm', help='the stage 1 model')
parser.add_argument('--num_shot', type=str, default='50', help='the number of shot')
parser.add_argument('--random_aug', action="store_true", default=False, help='random_aug?')

#### add mask
parser.add_argument('--activat_type', type=str, default='sigmoid', help='type of activation')
parser.add_argument('--opt_mask', action="store_true", default=False, help='also optimizer the mask?')
parser.add_argument('--pretrain_model', action="store_true", default=False, help='use pretrain model?')
parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')

# invariance
parser.add_argument('--inv', type=str, default='irm', help='type of invariant loss')
parser.add_argument('--inv_start', type=int, default=0, help='start epoch of inv loss')
parser.add_argument('--inv_weight', default=1., type=float, help='the weight of invariance')
parser.add_argument('--mlp', action="store_true", default=False, help='use mlp before the loss and feature?')

args = parser.parse_args()

best_acc1 = 0


'''
supervised contrastive loss
https://arxiv.org/abs/2004.11362
https://github.com/HobbitLong/SupContrast
'''
def info_nce_loss_supervised(features, batch_size, temperature=0.07, base_temperature=0.07, labels=None, choose_pos=None):
    ### features    bs * 2 * dim
    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().cuda()

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos

    if choose_pos is None:
        loss = loss.view(anchor_count, batch_size).mean()
    else:
        loss = loss.view(anchor_count, batch_size)[:, choose_pos].sum() / choose_pos.sum()

    return loss



class Model_Imagenet(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_Imagenet, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()
        # encoder
        model = Model_Imagenet()
        if os.path.isfile(pretrained_path):
            print("=> loading checkpoint '{}'".format(pretrained_path))
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            # state_dict = checkpoint['state_dict']
            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)
        else:
            print("=> no checkpoint found at '{}'".format(pretrained_path))
        self.f = model.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out



def main():
    if not os.path.exists('{}/{}'.format(args.save_root, args.name)):
        os.makedirs('{}/{}'.format(args.save_root, args.name))
    args.log_file = '{}/{}/eval_log.txt'.format(args.save_root, args.name)


    #################### Model ######################
    model_base = Net(num_class=args.class_num, pretrained_path=args.pretrain_path)
    import copy
    ft_fc = copy.deepcopy(model_base.fc)
    model_base.fc = nn.Identity()

    mask_layer = torch.rand(ft_fc.weight.size(1),)
    model = utils.ResNet_ft_eqinv(model_base, ft_fc, mask_layer=mask_layer, args=args)
    model = torch.nn.DataParallel(model).cuda()


    ######## define loss function (criterion) and optimizer
    init_lr = args.lr * args.batch_size / 256
    print('lr scale to %.2f' %(init_lr))
    criterion = nn.CrossEntropyLoss().cuda()
    if args.adam:
        init_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    ### prepare few shot dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.random_aug:
        train_transform_hard = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize, ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])


    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ])


    images = utils.Imagenet_idx(root=args.data+'/val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(images, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_images = utils.Imagenet_idx(root=args.data+'/testgt', transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    if args.random_aug:
        train_images = utils.Imagenet_idx_pair_transformone(root=args.data + '/train', transform_simple=train_transform, transform_hard=train_transform_hard)
    else:
        train_images = utils.Imagenet_idx_pair(root=args.data+'/train', transform=train_transform)
    memory_images = utils.Imagenet_idx(root=args.data + '/train', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True)
    memory_loader = torch.utils.data.DataLoader(memory_images, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    activation_map = utils.activation_map(args.activat_type)


    if args.evaluate:
        print('eval on vipriors val data')
        validate(val_loader, model, criterion, args, epoch=-1)
        print('eval on vipriors test data')
        validate(test_loader, model, criterion, args, epoch=-1)
        return


    #### Process Cluster
    assert args.stage1_model == 'ipirm'
    assert args.num_shot in ['10', '20', '50']

    if not os.path.exists('misc/env_ref_set_vipriors{}_rn50_{}_pretrained'.format(args.num_shot, args.stage1_model)):
        print('no cluster file, thus first process...')
        env_ref_set = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=0.1, anchor_class=None, class_debias_logits=True)
        torch.save(env_ref_set, 'misc/env_ref_set_vipriors{}_rn50_{}_pretrained'.format(args.num_shot, args.stage1_model))
    else:
        env_ref_set = torch.load('misc/env_ref_set_vipriors{}_rn50_{}_pretrained'.format(args.num_shot, args.stage1_model))


    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_env(train_loader, model, activation_map, env_ref_set, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))

    if args.opt_mask:
        torch.save(model.module.mask_layer, '{}/{}/mask_layer_opt'.format(args.save_root, args.name))

    utils.write_log('\nThe best accuracy: {}'.format(best_acc1), args.log_file, print_=True)
    utils.write_log('\nStart to test on Test Set', args.log_file, print_=True)
    acc1_test = validate(test_loader, model, criterion, args, epoch)




def train_env(train_loader, model, activation_map, env_ref_set, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_cont = AverageMeter('Loss_Cont', ':.4e')
    losses_inv = AverageMeter('Loss_Inv', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('LR', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_cont, losses_inv, top1, top5, LR],
        prefix="Epoch: [{}]".format(epoch),
        log_file=args.log_file)

    # switch to train mode
    model.train()

    all_sample_num = len(train_loader.dataset)

    end = time.time()
    for i, training_items in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.random_aug:
            images1, images2, images1_hard, images2_hard, target, images_idx = training_items
            images1_hard, images2_hard = images1_hard.cuda(non_blocking=True), images2_hard.cuda(non_blocking=True)
        else:
            images1, images2, target, images_idx = training_items
        images1, images2 = images1.cuda(non_blocking=True), images2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        masked_feature1, masked_feature_inv1, output1 = model(images1, return_masked_feature=True)
        masked_feature2, masked_feature_inv2, output2 = model(images2, return_masked_feature=True)
        if args.random_aug:
            output_hard = model(torch.cat([images1_hard, images2_hard]))

        masked_feature_for_globalcont, masked_feature, output = torch.cat([masked_feature1, masked_feature2], dim=0), torch.cat([masked_feature_inv1, masked_feature_inv2], dim=0), torch.cat([output1, output2], dim=0)
        target, images_idx = torch.cat([target, target], dim=0), torch.cat([images_idx, images_idx], dim=0)

        if args.inv_weight > 0:
            # compute env for different class
            env_nll, env_cont_nll, env_pen, temp_pen = [], [], [], []
            for class_idx in range(args.class_num):

                mask_pos = target==class_idx # choose the specifc positive samples
                if mask_pos.sum() == 0: # batch no current class
                    continue

                output_pos, target_num_pos, masked_feature_pos = output[mask_pos], target[mask_pos], masked_feature[mask_pos] # get positive and negative samples
                output_neg, images_idx_neg, target_num_neg, masked_feature_neg = output[~mask_pos], images_idx[~mask_pos], target[~mask_pos], masked_feature[~mask_pos]

                # generate the env lookup table
                env_ref_set_class = env_ref_set[class_idx]
                all_samples_env_table = torch.zeros(all_sample_num, len(env_ref_set_class))
                for env_idx in range(len(env_ref_set_class)):
                    all_samples_env_table[env_ref_set_class[env_idx], env_idx] = 1  # set samples according to current subset to 1

                # traversal different env
                for env_idx in range(len(env_ref_set_class)): # split the negative samples
                    output_neg_env, target_num_neg_env, masked_feature_neg_env = utils_cluster.assign_samples([output_neg, target_num_neg, masked_feature_neg], images_idx_neg, all_samples_env_table, env_idx)
                    output_env, target_num_env, masked_feature_env = torch.cat([output_pos, output_neg_env], dim=0), torch.cat([target_num_pos, target_num_neg_env], dim=0), torch.cat([masked_feature_pos, masked_feature_neg_env], dim=0)
                    masked_feature_env_norm = F.normalize(masked_feature_env, dim=-1)
                    cont_loss_env = args.cont_weight * info_nce_loss_supervised(masked_feature_env_norm.unsqueeze(1), masked_feature_env_norm.size(0), temperature=args.temperature, labels=target_num_env, choose_pos=target_num_env==class_idx)

                    env_nll.append(criterion(output_env, target_num_env))
                    temp_pen.append(cont_loss_env)

                env_pen.append(torch.var(torch.stack(temp_pen)))
                temp_pen = []


            # Invariance Term
            inv_weight = args.inv_weight if epoch >= args.inv_start else 0.
            assert args.inv == 'rex'
            rex_penalty = sum(env_pen) / len(env_pen)
            loss_inv = inv_weight * rex_penalty

        else:
            loss_inv = torch.Tensor([0.]).cuda()


        # ERM loss
        if args.random_aug:
            loss_erm = criterion(output_hard, target)
        else:
            loss_erm = criterion(output, target)
        masked_feature_for_globalcont_norm = F.normalize(masked_feature_for_globalcont, dim=-1)
        loss_cont = args.cont_weight * info_nce_loss_supervised(masked_feature_for_globalcont_norm.unsqueeze(1), masked_feature_for_globalcont_norm.size(0), temperature=args.temperature, labels=target)


        loss_all = loss_erm + loss_cont + loss_inv


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_erm.item(), images1.size(0)+images2.size(0))
        losses_cont.update(loss_cont.item(), images1.size(0)+images2.size(0))
        losses_inv.update(loss_inv.item(), images1.size(0)+images2.size(0))
        top1.update(acc1[0], images1.size(0)+images2.size(0))
        top5.update(acc5[0], images1.size(0)+images2.size(0))
        LR.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        log_file=args.log_file)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, images_idx) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary(epoch)

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}/model_best.pth.tar'.format(args.save_root, args.name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file='eval_log.txt'):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


    def display_summary(self, epoch):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

        utils.write_log('Test Epoch {} '.format(epoch) + ' '.join(entries), self.log_file, print_=False)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




if __name__ == '__main__':
    main()

