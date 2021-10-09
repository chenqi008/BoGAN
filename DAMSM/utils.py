import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

import cv2

# from miscc.config import cfg
from config import cfg

# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.load_default().font
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_len, att_sze, input_channels,
                       lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    # if att_sze == 17:
    #     vis_size = att_sze * 16
    # else:
    #     vis_size = real_imgs.size(2)
    vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    # print(attn_maps[0].shape)
    # assert False

    bUpdate = 1
    for k in range(16):
        for i in range(num):
            # attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
            attn = attn_maps[i].cpu().view(1, -1, att_len, att_sze, att_sze)
            # --> 1 x 1 x 17 x 17
            attn_max = attn.max(dim=1, keepdim=True)

            # print(attn_max[0].shape)
            # assert False
            
            # => 1x15x16x6x6
            attn = torch.cat([attn_max[0], attn], 1)

            # print(att_sze)
            # print(att_len)
            # assert False
            #
            # attn = attn.view(-1, 1, att_sze, att_sze)
            # attn = attn.repeat(1, 3, 1, 1).data.numpy()
            attn = attn.view(-1, 1, att_len, att_sze, att_sze)
            attn = attn.repeat(1, 3, 1, 1, 1).data.numpy()
            # # n x c x h x w --> n x h x w x c
            # attn = np.transpose(attn, (0, 2, 3, 1))
            # n x c x l x h x w --> n x l x h x w x c
            attn = np.transpose(attn, (0, 2, 3, 4, 1))
            num_attn = attn.shape[0]
            #
            img = real_imgs[i]

            # print(type(img[0][0][0]))
            # test_img = Image.fromarray(np.uint8(img[:, :, 0:3]))
            # test_img.save('test.png')
            # assert False

            # 48x48x48
            # print(img.shape)
            # assert False

            if lr_imgs is None:
                lrI = img
            else:
                lrI = lr_imgs[i]
            # row = [lrI, middle_pad]
            # row_merge = [img, middle_pad]
            if input_channels==1:
                # row = [lrI[:,:,[0]].repeat(3, axis=2), middle_pad]
                # row_merge = [img[:,:,[0]].repeat(3, axis=2), middle_pad]
                row_merge_global = [img[:,:,[0]].repeat(3, axis=2), middle_pad]
            elif input_channels==3:
                # row = [lrI[:,:,:3], middle_pad]
                # row_merge = [img[:,:,:3], middle_pad]
                row_merge_global = [img[:,:,:3], middle_pad]
            row_beforeNorm = []
            minVglobal, maxVglobal = 1, 0

            for j in range(num_attn):
                one_map = attn[j]

                # print(one_map)
                # assert False

                # if (vis_size // att_sze) > 1:
                #     one_map = \
                #         skimage.transform.pyramid_expand(one_map, sigma=20,
                #                                          upscale=vis_size // att_sze)
                one_map_temp = []
                if (vis_size // att_sze) > 1:
                    for k in xrange(len(one_map)):
                        temp = \
                            skimage.transform.pyramid_expand(one_map[k], sigma=30,
                                                             upscale=vis_size // att_sze,
                                                             mode='reflect', cval=0.0)
                        # for aa in xrange(48):
                        #     print('===========aa=%s==========='%(aa))
                        #     for bb in xrange(48):
                        #         print(temp[aa,bb,0])
                        # assert False
                        # temp 48x48x3
                        one_map_temp.append(temp)

                # one_map_temp 16x48x48x3
                one_map = np.asarray(one_map_temp)

                # print(one_map.shape)
                # assert False


                row_beforeNorm.append(one_map)
                minV = one_map.min()
                maxV = one_map.max()
                if minVglobal > minV:
                    minVglobal = minV
                if maxVglobal < maxV:
                    maxVglobal = maxV

                # 16x48x48x3
                # print(one_map.shape)
                # assert False

            # row_beforeNorm nx16x48x48x3
            row_merge = row_merge_global
            for j in range(seq_len + 1):
                if j < num_attn:
                    one_map = row_beforeNorm[j]
                    # one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                    # print(one_map)
                    # assert False
                    # one_map *= 255

                    #
                    # PIL_im = Image.fromarray(np.uint8(img))
                    # PIL_att = Image.fromarray(np.uint8(one_map))
                    # if input_channels==1:
                    #     PIL_im = Image.fromarray(np.uint8(img[:,:,[j]].repeat(3, axis=2)))
                    # elif input_channels==3:
                    #     PIL_im = Image.fromarray(np.uint8(img[:,:,j*3:(j+1)*3]))
                    if input_channels==1:
                        PIL_im = Image.fromarray(np.uint8(img[:,:,[k]].repeat(3, axis=2)))
                    elif input_channels==3:
                        PIL_im = Image.fromarray(np.uint8(img[:,:,k*3:(k+1)*3]))

                    # PIL_att = Image.fromarray(np.uint8(one_map[j,:,:,:]))

                    # curr_one_map = np.uint8(one_map[j,:,:,:])
                    curr_one_map = np.float32(one_map[j,:,:,:])
                    # curr_max = np.max(curr_one_map)
                    # curr_min = np.min(curr_one_map)
                    curr_max = np.max(one_map)
                    curr_min = np.min(one_map)
                    curr_one_map = (curr_one_map - curr_min) / (curr_max - curr_min)

                    curr_one_map *= 255
                    curr_one_map = curr_one_map.astype(np.uint8)

                    cv2_att = cv2.applyColorMap(curr_one_map, cv2.COLORMAP_JET)
                    cv2_att = np.float32(cv2_att)/255.0

                    cv2_im = np.asarray(PIL_im)/255.0


                    cv2_im_att = cv2_att*0.9+cv2_im

                    cv2_im_att = cv2_im_att / np.max(cv2_im_att)

                    merged = np.uint8(255*cv2_im_att)

                    # row.append(one_map[j,:,:,:])
                else:
                    one_map = post_pad
                    merged = post_pad
                    # row.append(one_map)

                # row.append(middle_pad)
                # row.append(one_map)
                # row.append(middle_pad)
                #
                row_merge.append(merged)
                row_merge.append(middle_pad)

            # print(row[2].shape)
            # assert False
            # row = np.concatenate(row, 1)
            row_merge = np.concatenate(row_merge, 1)
            txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
            # if txt.shape[1] != row.shape[1]:
            #     print('txt', txt.shape, 'row', row.shape)
            #     bUpdate = 0
            #     break
            if txt.shape[1] != row_merge.shape[1]:
                print('txt', txt.shape, 'row_merge', row_merge.shape)
                bUpdate = 0
                break
            # row = np.concatenate([txt, row, row_merge], 0)
            row = np.concatenate([txt, row_merge], 0)
            img_set.append(row)

    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
