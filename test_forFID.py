import torch
from torch import optim
from torch import nn
import os
import numpy as np
import random

# decide random seed
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# manualSeed = 123
# random.seed(manualSeed)
# np.random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)


# from model import _G, _D, _D_frame_motion
from model import _G
# import dataset
import torchvision.transforms as transforms
import vutils

# from text_encoder import Text2Embedding

from DAMSM.losses import words_loss, sent_loss, KL_loss
from DAMSM.model_DAMSM import RNN_ENCODER, CNN_ENCODER
from DAMSM.datasets import TextDataset, prepare_data, prepare_data_real_wrong

import time
import math


# This is a helper function to print time elapsed and estimated time remaining given the current time and progress.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent==0:
        es = s
    else:
        es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def generateZ(args):
    if args.z_dis == "norm":
        # Z = torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33).cuda()
        Z = torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).cuda()
    elif args.z_dis == "uni":
        Z = torch.rand(args.batch_size, args.z_size).cuda()
    else:
        print("z_dist is not normal or uniform")
    return Z


# @profile
def test(args):
    # set devices
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(args.gpu))


    # Get data loader ##################################################
    image_transform = transforms.Compose([transforms.Resize(args.imageSize)])
    dataset = TextDataset(args.dataroot, 'train',
                          base_size=args.imageSize,
                          transform=image_transform,
                          input_channels=args.input_channels,
                          image_type=args.image_type)
    dset_loaders = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True,
        shuffle=True, num_workers=args.workers)


    # ============== #
    # Define D and G #
    # ============== #
    # D = _D(args)
    # D_frame_motion = _D_frame_motion(args)
    G = _G(args)
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=args.hidden_size, batch_size=args.batch_size)

    if args.cuda:
        print("using cuda")
        if args.gpu_num>1:
            device_ids = range(args.gpu, args.gpu+args.gpu_num)
            # D = nn.DataParallel(D, device_ids = device_ids)
            # D_frame_motion = nn.DataParallel(D_frame_motion, device_ids = device_ids)
            G = nn.DataParallel(G, device_ids = device_ids)
            # text_encoder = nn.DataParallel(text_encoder, device_ids = device_ids)
            # image_encoder = nn.DataParallel(image_encoder, device_ids = device_ids)
        # D.cuda()
        # D_frame_motion.cuda()
        G.cuda()
        text_encoder.cuda()
        # image_encoder.cuda()
        # criterion = criterion.cuda()
        # criterion_l1 = criterion_l1.cuda()
        # criterion_l2 = criterion_l2.cuda()


    if args.checkpoint_G != '':
        G.load_state_dict(torch.load(args.checkpoint_G, map_location='cpu'))

    # ================================================== #
    # Load text and image encoder and fix the parameters #
    # ================================================== #
    # text_encoder = RNN_ENCODER(dataset.n_words, nhidden=args.hidden_size, batch_size=args.batch_size)
    # image_encoder = CNN_ENCODER(args.hidden_size)

    if args.checkpoint_text_encoder != '':
        text_encoder.load_state_dict(torch.load(args.checkpoint_text_encoder))
    # if args.checkpoint_image_encoder != '':
    #     image_encoder.load_state_dict(torch.load(args.checkpoint_image_encoder))

    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from: %s'%args.checkpoint_text_encoder)
    
    text_encoder.eval()
    G.eval()

    # criterion of update
    # D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    # D_solver_frame = optim.Adam(D_frame_motion.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)


    if args.cls:
        print("using cls")
    if args.A:
        print("using A")
    # if args.C:
    #     print("using C")
    if args.video_loss:
        print("using video discriminator")
    if args.frame_motion_loss:
        print("using frame and motion discriminator")

    # estimate time
    start = time.time()

    counter_folder = 0
    # iteration
    for epoch in range(args.n_epochs):
        for i, data in enumerate(dset_loaders, 0):
            if i%10==0:
                print('Epoch[{}][{}/{}] Time: {}'.format(epoch, i, len(dset_loaders),
                    timeSince(start, float(epoch*len(dset_loaders)+i) / float(args.n_epochs*len(dset_loaders)))))

            # if (i+1)%5==0:
            #     break

            X, captions, cap_lens, class_ids, keys, \
            X_wrong, caps_wrong, cap_len_wrong, cls_id_wrong, key_wrong \
            = prepare_data_real_wrong(data)
            X = X[0]
            X_wrong = X_wrong[0]


            hidden = text_encoder.init_hidden()

            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            # ==================== #
            # Generate fake images #
            # ==================== #
            # generate input noize Z (size: batch_size x 100)
            Z = generateZ(args)
            # Z = torch.cat([Z, embedding], 1)

            fake, mu, logvar = G(Z, sent_emb)

            # save images and sentence
            if (epoch + 1) % args.image_save_step == 0:
                fid_image_path = os.path.join(args.output_dir, args.fid_fake_foldername, "images")
                if not os.path.exists(fid_image_path):
                    os.makedirs(fid_image_path)
                counter_folder = vutils.save_image_forfinalFID(fake, None, 
                    normalize=True, pad_value=1, input_channels=args.input_channels, 
                    imageSize=args.imageSize, fid_image_path=fid_image_path,
                    counter_folder=counter_folder)




