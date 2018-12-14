import os
import re
from collections import defaultdict
from random import shuffle
import random
import shutil
random.seed(22)

img_dir = '/home/teddy/ut-zap50k-images'
img_dir_1 = '/home/teddy/ut-zap50k-images-square'
save_dir = '/home/teddy/shoes'

regex_0 = re.compile(r'\sand\s')
regex_1 = re.compile(r'\s+')
label_img_dic = defaultdict(list)
ratio = 0.1

for root, dirs, files in os.walk(img_dir, topdown=True):
    img_list = [i for i in files if 'jpg' in i]
    if len(img_list) > 0:
        img_list = [os.path.join(root, i) for i in img_list]
        items = root.split('/')
        label = items[-3] + '_' + items[-2]
        label = regex_0.sub('_', label)
        label = regex_1.sub('_', label)

        label_img_dic[label].extend(img_list)

im_dic = defaultdict(dict)
for key in label_img_dic:
    img_list = label_img_dic[key]
    shuffle(img_list)
    val_num = int(len(img_list) * ratio)
    im_dic['train'][key] = img_list[val_num:]
    im_dic['val'][key] = img_list[:val_num]

regex_2 = re.compile('images')
for key in im_dic['train']:
    count = 0
    for phase in ['train', 'val']:
        sub_dir = os.path.join(save_dir, phase, key)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        im_list = im_dic[phase][key]
        for path in im_list:
            shutil.copy(path, os.path.join(sub_dir, '{}.jpg'.format(count)))
            count += 1
            shutil.copy(path, os.path.join(regex_2.sub('images-square', sub_dir), '{}.jpg'.format(count)))
            count += 1
