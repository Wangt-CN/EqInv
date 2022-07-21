import os
import shutil
import random

num_shot = 10
vipriors_dataset_path = '/your_path_to/imagenet_50' # the path of the source file
target_path = '/your_path_to/imagenet_{}'.format(num_shot) # the path to generate file


###
# random.seed(0)
choose_idx = random.sample(range(50), num_shot) # randomly choose the image idx


## produce train
dataset_class_list = os.listdir('{}/train'.format(vipriors_dataset_path))
dataset_class_list.sort()

for class_name in dataset_class_list:
    vipriors_dataset_class_path = '{}/train/{}'.format(vipriors_dataset_path, class_name)
    target_dataset_class_path = '{}/train/{}'.format(target_path, class_name)
    os.makedirs(target_dataset_class_path)

    dataset_file_list = os.listdir(vipriors_dataset_class_path)
    dataset_file_list.sort()
    for class_choose_idx in choose_idx:
        src_pth = os.path.join(vipriors_dataset_class_path, dataset_file_list[class_choose_idx])
        tgt_pth = os.path.join(target_dataset_class_path, dataset_file_list[class_choose_idx])
        shutil.copyfile(src_pth, tgt_pth)



# validation and test set follow the original vipriors-50 data