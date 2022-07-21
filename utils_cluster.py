import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from torch import nn, optim, autograd



def cal_cosine_distance(net, memory_data_loader, c, temperature, anchor_class=None, class_debias_logits=False, mask=None):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank, idx_bank = 0.0, 0.0, 0, [], [], []

    with torch.no_grad():
        # generate feature bank
        for images, target, images_idx in tqdm(memory_data_loader, desc='Feature extracting'):
            images = images.cuda(non_blocking=True)
            feature = net(images, return_feature=True)
            if mask is not None:
                feature = mask * feature
            feature_bank.append(F.normalize(feature, dim=-1))
            target_bank.append(target)
            idx_bank.append(images_idx)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels_digit = torch.cat(target_bank, dim=0).contiguous()
        idx_bank = torch.cat(idx_bank, dim=0).contiguous()


    if anchor_class is None:
        anchor_class_set = range(c)
    else:
        anchor_class_set = [anchor_class]

    env_set = {}
    for anchor_class_ in anchor_class_set:
        print('\rcosine distance to anchor class {}'.format(anchor_class_), end='')
        anchor_mask = feature_labels_digit == anchor_class_
        candidate_mask = ~anchor_mask

        anchor_feature, anchor_idx = feature_bank[:, anchor_mask], idx_bank[anchor_mask]
        candidate_feature = feature_bank[:, candidate_mask].t()
        candidate_labels_digit = feature_labels_digit[candidate_mask]

        # loop the candidate feature
        sim_all = []
        candidate_dataloader = data.DataLoader(SampleFeature(candidate_feature), batch_size=1024, shuffle=False, num_workers=0)
        for candidate_feature_batch in candidate_dataloader:
            sim_matrix = torch.mm(candidate_feature_batch, anchor_feature)
            # sim_matrix = (sim_matrix / temperature).exp()
            sim_batch = sim_matrix.mean(dim=-1)
            sim_all.append(sim_batch)
        sim_all = torch.cat(sim_all, dim=0).contiguous()


        if class_debias_logits: # calculate a class-wise debias logits to remove the digits similarity effect
            class_debias_logits_weight = torch.zeros(c).to(sim_all.device)
            for iii in range(c):
                if iii == anchor_class_:
                    class_debias_logits_weight[iii] = 1.
                    continue
                find_idx = torch.where(candidate_labels_digit == iii)[0]
                class_debias_logits_weight[iii] = sim_all[find_idx].mean()
            sim_all_debias_logits = class_debias_logits_weight[candidate_labels_digit]
            sim_all -= sim_all_debias_logits


        sim_sort = torch.argsort(sim_all, descending=True)
        candidate_idx_sort = idx_bank[candidate_mask][sim_sort]

        # import pdb
        # pdb.set_trace()

        env_set[anchor_class_] = torch.chunk(candidate_idx_sort, 2) # 2 environments

    return env_set




class SampleFeature(data.Dataset):
    def __init__(self, feature_bank):
        """Initialize and preprocess the dataset."""
        self.feature_bank = feature_bank


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        feature = self.feature_bank[index]

        return feature

    def __len__(self):
        """Return the number of images."""
        return self.feature_bank.size(0)



def penalty(logits, y, loss_function):
    # scale = torch.tensor(1.).cuda().requires_grad_()
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def penalty_bce(logits, y, loss_function):
    # scale = torch.tensor(1.).cuda().requires_grad_()
    scale = torch.ones((logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def assign_samples(items, idxs, split, env_idx):
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return [i[select_idx] for i in items]
