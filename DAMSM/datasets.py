from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

try:
    from config import cfg
except:
    from DAMSM.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import os

def prepare_data(data):
    # imgs, captions, captions_lens, class_ids, keys = data
    imgs, captions, captions_lens, class_ids, keys, _, _, _, _, _ = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]


def prepare_data_real_wrong(data):
    imgs, captions, captions_lens, class_ids, keys, \
    imgs_wrong, captions_wrong, captions_lens_wrong, class_ids_wrong, keys_wrong \
    = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    wrong_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        imgs_wrong[i] = imgs_wrong[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
            wrong_imgs.append(Variable(imgs_wrong[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
            wrong_imgs.append(Variable(imgs_wrong[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()

    captions_wrong = captions_wrong[sorted_cap_indices].squeeze()
    class_ids_wrong = class_ids_wrong[sorted_cap_indices].numpy()
    sorted_cap_lens_wrong = captions_lens_wrong[sorted_cap_indices]

    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    keys_wrong = [keys_wrong[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        captions_wrong = Variable(captions_wrong).cuda()
        sorted_cap_lens_wrong = Variable(sorted_cap_lens_wrong).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        captions_wrong = Variable(captions_wrong)
        sorted_cap_lens_wrong = Variable(sorted_cap_lens_wrong)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys,
            wrong_imgs, captions_wrong, sorted_cap_lens_wrong, class_ids_wrong, keys_wrong]



def get_imgs(img_path, imsize,
             transform=None, normalize=None, input_channels=3):
    if input_channels==1:
        img = Image.open(img_path).convert('L')
    elif input_channels==3:
        img = Image.open(img_path).convert('RGB')
    else:
        assert False
    
    width, height = img.size

    if transform is not None:
        img = transform(img)

    return normalize(img)



class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=48, transform=None,
                 target_transform=None, input_channels=3,
                 image_type='jpg'):
        self.image_type = image_type
        self.input_channels = input_channels
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        # self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.embeddings_num = 1

        self.imsize = []
        # for i in range(cfg.TREE.BRANCH_NUM):
        #     self.imsize.append(base_size)
        #     base_size = base_size * 2
        self.imsize.append(base_size)

        self.data = []
        self.data_dir = data_dir

        self.split = split

        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        print("number of class in dataset: ", len(self.class_id))
        # if split == 'train':
        #     self.class_id = range(60000)
        # else:
        #     self.class_id = range(10000)
        self.number_example = len(self.filenames)


    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            # cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            if self.split=='train':
                cap_path = '%s/train_text/%s.txt' % (data_dir, filenames[i])
            else:
                cap_path = '%s/test_text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_wrong_image(self, index, num_index, caps_real):
        # choose another index of sample
        index_wrong = random.randint(0, num_index-1)

        while index_wrong==index:
            index_wrong = random.randint(0, num_index-1)

        # check whether the sentence is the same as the real samples'
        sent_ix_wrong = random.randint(0, self.embeddings_num)
        new_sent_ix_wrong = index_wrong * self.embeddings_num + sent_ix_wrong
        caps_wrong, cap_len_wrong = self.get_caption(new_sent_ix_wrong)

        # while torch.equal(caps_real, caps_wrong):
        while (caps_real==caps_wrong).all():
            # choose another index of sample
            index_wrong = random.randint(0, num_index-1)
            while index_wrong==index:
                index_wrong = random.randint(0, num_index-1)
            # check whether the sentence is the same as the real samples'
            sent_ix_wrong = random.randint(0, self.embeddings_num)
            new_sent_ix_wrong = index_wrong * self.embeddings_num + sent_ix_wrong
            caps_wrong, cap_len_wrong = self.get_caption(new_sent_ix_wrong)

        # get wrong image
        key_wrong = self.filenames[index_wrong]
        cls_id_wrong = self.class_id[index_wrong]
        #
        if self.split == 'train':
            img_name = '%s/train_image/%s.%s' % (self.data_dir, key_wrong, self.image_type)
        else:
            img_name = '%s/test_image/%s.%s' % (self.data_dir, key_wrong, self.image_type)

        imgs_wrong = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm, input_channels=self.input_channels)
        for i in range(1, 16):
            temp = get_imgs("%s%02d.%s"%(img_name[:-6], i, self.image_type), self.imsize, self.transform, normalize=self.norm, input_channels=self.input_channels)
            imgs_wrong = torch.cat([imgs_wrong, temp], 0)
        imgs_wrong = [imgs_wrong]

        return imgs_wrong, caps_wrong, cap_len_wrong, cls_id_wrong, key_wrong


    def __getitem__(self, index):
        # get real image
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.split == 'train':
            img_name = '%s/train_image/%s.%s' % (self.data_dir, key, self.image_type)
        else:
            img_name = '%s/test_image/%s.%s' % (self.data_dir, key, self.image_type)

        imgs = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm, input_channels=self.input_channels)
        for i in range(1, 16):
            temp = get_imgs("%s%02d.%s"%(img_name[:-6], i, self.image_type), self.imsize, self.transform, normalize=self.norm, input_channels=self.input_channels)
            imgs = torch.cat([imgs, temp], 0)
        imgs = [imgs]

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        # get wrong image
        imgs_wrong, caps_wrong, cap_len_wrong, cls_id_wrong, key_wrong \
         = self.get_wrong_image(index, len(self.filenames), caps)

        return imgs, caps, cap_len, cls_id, key, \
         imgs_wrong, caps_wrong, cap_len_wrong, cls_id_wrong, key_wrong


    def __len__(self):
        return len(self.filenames)
