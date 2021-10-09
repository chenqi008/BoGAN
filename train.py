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


from model import _G, _D, _D_frame_motion

import torchvision.transforms as transforms
import vutils

from DAMSM.losses import words_loss, sent_loss, KL_loss
from DAMSM.model_DAMSM import RNN_ENCODER, CNN_ENCODER
from DAMSM.datasets import TextDataset, prepare_data, prepare_data_real_wrong

import time
import math

from calculate_fid_pytorch.fid import fid_score


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
def train(args):
    # set devices
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(args.gpu))


    # Get data loader ##################################################
    image_transform = transforms.Compose([transforms.Resize(args.imageSize),
        transforms.CenterCrop(args.imageSize)])
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
    D = _D(args)
    D_frame_motion = _D_frame_motion(args)
    G = _G(args)

    if args.cuda:
        # print("using cuda")
        if args.gpu_num>0:
            device_ids = range(args.gpu, args.gpu+args.gpu_num)
            D = nn.DataParallel(D, device_ids = device_ids)
            D_frame_motion = nn.DataParallel(D_frame_motion, device_ids = device_ids)
            G = nn.DataParallel(G, device_ids = device_ids)

    if args.checkpoint_D != '':
        D.load_state_dict(torch.load(args.checkpoint_D))
    if args.checkpoint_frame_motion_D != '':
        D_frame_motion.load_state_dict(torch.load(args.checkpoint_frame_motion_D))
    if args.checkpoint_G != '':
        G.load_state_dict(torch.load(args.checkpoint_G))

    # ================================================== #
    # Load text and image encoder and fix the parameters #
    # ================================================== #
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=args.hidden_size, batch_size=args.batch_size)
    image_encoder = CNN_ENCODER(args.hidden_size, args.input_channels)

    if args.checkpoint_text_encoder != '':
        text_encoder.load_state_dict(torch.load(args.checkpoint_text_encoder))
    if args.checkpoint_image_encoder != '':
        image_encoder.load_state_dict(torch.load(args.checkpoint_image_encoder))

    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from: %s'%args.checkpoint_text_encoder)
    text_encoder.eval()

    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from: %s'%args.checkpoint_image_encoder)
    image_encoder.eval()


    # criterion of update
    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    D_solver_frame = optim.Adam(D_frame_motion.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)


    # loss
    criterion = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    # criterion_l2 = nn.MSELoss(size_average=False)

    if args.cuda:
        print("using cuda")
        # if args.gpu_num>1:
        #     device_ids = range(args.gpu, args.gpu+args.gpu_num)
        #     D = nn.DataParallel(D, device_ids = device_ids)
        #     D_frame_motion = nn.DataParallel(D_frame_motion, device_ids = device_ids)
        #     G = nn.DataParallel(G, device_ids = device_ids)
            # text_encoder = nn.DataParallel(text_encoder, device_ids = device_ids)
            # image_encoder = nn.DataParallel(image_encoder, device_ids = device_ids)
        D.cuda()
        D_frame_motion.cuda()
        G.cuda()
        text_encoder.cuda()
        image_encoder.cuda()
        criterion = criterion.cuda()
        criterion_l1 = criterion_l1.cuda()
        # criterion_l2 = criterion_l2.cuda()

    # # print the parameters of model
    # params=G.state_dict()
    # for k, v in params.items():
    #     print(k)
    # print(params['deconv.0.weight'])

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

    # # generator labels
    # if args.cuda:
    #     real_labels_G = torch.ones(args.batch_size).cuda()
    #     real_labels_G_frame = torch.ones(args.batch_size*args.frame_num).cuda()
    #     real_labels_G_motion = torch.ones(args.batch_size*(args.frame_num-1)).cuda()
    # else:
    #     real_labels_G = torch.ones(args.batch_size)
    #     real_labels_G_frame = torch.ones(args.batch_size*args.frame_num)
    #     real_labels_G_motion = torch.ones(args.batch_size*(args.frame_num-1))

    # video labels
    if args.soft_label:
        if args.cuda:
            real_labels = torch.Tensor(args.batch_size).uniform_(0.7, 1.2).cuda()
            fake_labels = torch.Tensor(args.batch_size).uniform_(0, 0.3).cuda()
        else:
            real_labels = torch.Tensor(args.batch_size).uniform_(0.7, 1.2)
            fake_labels = torch.Tensor(args.batch_size).uniform_(0, 0.3)
    else:
        if args.cuda:
            real_labels = torch.ones(args.batch_size).cuda()
            fake_labels = torch.zeros(args.batch_size).cuda()
        else:
            real_labels = torch.ones(args.batch_size)
            fake_labels = torch.zeros(args.batch_size)

    # frame labels
    if args.soft_label:
        if args.cuda:
            real_labels_frame = torch.Tensor(args.batch_size*args.frame_num).uniform_(0.7, 1.2).cuda()
            fake_labels_frame = torch.Tensor(args.batch_size*args.frame_num).uniform_(0, 0.3).cuda()
        else:
            real_labels_frame = torch.Tensor(args.batch_size*args.frame_num).uniform_(0.7, 1.2)
            fake_labels_frame = torch.Tensor(args.batch_size*args.frame_num).uniform_(0, 0.3)
    else:
        if args.cuda:
            real_labels_frame = torch.ones(args.batch_size*args.frame_num).cuda()
            fake_labels_frame = torch.zeros(args.batch_size*args.frame_num).cuda()
        else:
            real_labels_frame = torch.ones(args.batch_size*args.frame_num)
            fake_labels_frame = torch.zeros(args.batch_size*args.frame_num)

    # motion labels
    if args.A:
        if args.soft_label:
            if args.cuda:
                real_labels_motion = torch.Tensor(args.batch_size*(args.frame_num-1)).uniform_(0.7, 1.2).cuda()
                fake_labels_motion = torch.Tensor(args.batch_size*(args.frame_num-1)).uniform_(0, 0.3).cuda()
            else:
                real_labels_motion = torch.Tensor(args.batch_size*(args.frame_num-1)).uniform_(0.7, 1.2)
                fake_labels_motion = torch.Tensor(args.batch_size*(args.frame_num-1)).uniform_(0, 0.3)
        else:
            if args.cuda:
                real_labels_motion = torch.ones(args.batch_size*(args.frame_num-1)).cuda()
                fake_labels_motion = torch.zeros(args.batch_size*(args.frame_num-1)).cuda()
            else:
                real_labels_motion = torch.ones(args.batch_size*(args.frame_num-1))
                fake_labels_motion = torch.zeros(args.batch_size*(args.frame_num-1))

    # matching labels
    if args.cuda:
        match_labels = torch.LongTensor(range(args.batch_size)).cuda()
    else:
        match_labels = torch.LongTensor(range(args.batch_size))


    best_fid = 10000.0
    best_epoch = 0
    # iteration
    for epoch in range(args.n_epochs):
        for i, data in enumerate(dset_loaders, 0):
            if i%10==0:
                print('Epoch[{}][{}/{}] Time: {}'.format(epoch, i, len(dset_loaders),
                    timeSince(start, float(epoch*len(dset_loaders)+i) / float(args.n_epochs*len(dset_loaders)))))

            # ========================= #
            #  Prepare training data    #
            # ========================= #

            X, captions, cap_lens, class_ids, keys, \
            X_wrong, caps_wrong, cap_len_wrong, cls_id_wrong, key_wrong \
            = prepare_data_real_wrong(data)

            X = X[0]
            X_wrong = X_wrong[0]

            # if args.gpu_num > 1:
            #     hidden = text_encoder.module.init_hidden()
            # else:
            #     hidden = text_encoder.init_hidden()

            hidden = text_encoder.init_hidden()

            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            # mask = (captions==0)
            # num_words = words_embs.size(2)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]

            # ==================== #
            # Generate fake images #
            # ==================== #
            # generate input noize Z (size: batch_size x 100)
            Z = generateZ(args)

            fake, mu, logvar = G(Z, sent_emb)
            # fake, mu, logvar = G(Z, sent_emb, words_embs, mask)

            # ========================= #
            #  Train the discriminator  #
            # ========================= #

            # ====== video discriminator ====== #

            D.zero_grad()

            # real
            d_real = D(X, sent_emb)
            d_real_loss = criterion(d_real, real_labels)

            # wrong
            d_wrong = D(X_wrong, sent_emb)
            d_wrong_loss = criterion(d_wrong, fake_labels)

            # fake
            d_fake = D(fake, sent_emb)
            d_fake_loss = criterion(d_fake, fake_labels)

            # final
            d_loss = d_real_loss + d_fake_loss + d_wrong_loss

            # # tricks
            # d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            # d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            # d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))
            # # if d_total_acu <= args.d_thresh:
            d_loss.backward()
            D_solver.step()

            # ====== frame and motion discriminator ====== #

            D_frame_motion.zero_grad()

            # d_real_frame, d_real_motion = D_frame_motion(X, embedding)
            d_real_frame, d_real_motion = D_frame_motion(X, sent_emb)
            d_real_loss_frame = criterion(d_real_frame, real_labels_frame)
            d_real_loss_motion = criterion(d_real_motion, real_labels_motion)

            # d_wrong_frame, d_wrong_motion = D_frame_motion(X_wrong, embedding)
            d_wrong_frame, d_wrong_motion = D_frame_motion(X_wrong, sent_emb)
            d_wrong_loss_frame = criterion(d_wrong_frame, fake_labels_frame)
            d_wrong_loss_motion = criterion(d_wrong_motion, fake_labels_motion)

            # fake = G(Z)
            fake, mu, logvar = G(Z, sent_emb)
            # fake, mu, logvar = G(Z, sent_emb, words_embs, mask)

            # d_fake_frame, d_fake_motion = D_frame_motion(fake, embedding)
            d_fake_frame, d_fake_motion = D_frame_motion(fake, sent_emb)
            d_fake_loss_frame = criterion(d_fake_frame, fake_labels_frame)
            d_fake_loss_motion = criterion(d_fake_motion, fake_labels_motion)


            d_loss_frame = d_real_loss_frame + d_fake_loss_frame + d_wrong_loss_frame \
                            + d_real_loss_motion + d_wrong_loss_motion + d_fake_loss_motion

            # # tricks
            # d_real_acu_frame = torch.ge(d_real_frame.squeeze(), 0.5).float()
            # d_fake_acu_frame = torch.le(d_fake_frame.squeeze(), 0.5).float()
            # d_total_acu_frame = torch.mean(torch.cat((d_real_acu_frame, d_fake_acu_frame),0))
            # # if d_total_acu_frame <= args.d_thresh:
            d_loss_frame.backward()
            D_solver_frame.step()


            # ===================== #
            #  Train the generator  #
            # ===================== #

            G.zero_grad()

            # generate fake samples
            fake, mu, logvar = G(Z, sent_emb)
            # fake, mu, logvar = G(Z, sent_emb, words_embs, mask)

            # calculate the loss
            d_fake = D(fake, sent_emb)
            g_loss = criterion(d_fake, real_labels)
            # g_loss = criterion(d_fake, real_labels_G)

            # frame and motion
            d_fake_frame, d_fake_motion = D_frame_motion(fake, sent_emb)
            g_loss_frame = criterion(d_fake_frame, real_labels_frame)
            g_loss_motion = criterion(d_fake_motion, real_labels_motion)

            # # frame and motion
            # d_fake_frame, d_fake_motion = D_frame_motion(fake, sent_emb)
            # g_loss_frame = criterion(d_fake_frame, real_labels_G_frame)
            # g_loss_motion = criterion(d_fake_motion, real_labels_G_motion)

            # add to g_loss
            g_loss = g_loss + g_loss_frame + g_loss_motion

            # (DAMSM)
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake)

            # the loss of each word
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, args.batch_size)
            w_loss = (w_loss0 + w_loss1) * args.lamb

            # the loss of the whole sentence
            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                             match_labels, class_ids, args.batch_size)
            s_loss = (s_loss0 + s_loss1) * args.lamb

            # add to g_loss
            g_loss = g_loss + w_loss + s_loss

            # kl loss
            kl_loss = KL_loss(mu, logvar)

            # add to g_loss
            g_loss += kl_loss

            g_loss.backward()
            G_solver.step()


            # if i==3:
            #     # print the parameters of model
            #     params=G.state_dict()
            #     for k, v in params.items():
            #         print(k)
            #     print(params['deconv.0.weight'])
            #     assert False


        # =============== each epoch save model or save image =============== #
        if args.frame_motion_loss==True and args.video_loss==False:
            print('Epoch-{}; D_loss_frame:{:.4}, G_loss:{:.4}, D_lr:{:.4}'.format(
                epoch, d_loss_frame.item(), g_loss.item(), D_solver.state_dict()['param_groups'][0]["lr"]))
        elif args.frame_motion_loss==False and args.video_loss==True:
            print('Epoch-{}; D_loss_video:{:.4}, G_loss:{:.4}, D_lr:{:.4}'.format(
                epoch, d_loss.item(), g_loss.item(), D_solver.state_dict()['param_groups'][0]["lr"]))
        else:
            print('Epoch-{}; D_loss_video:{:.4}, D_loss_frame:{:.4}, G_loss:{:.4}, D_lr:{:.4}'.format(
                epoch, d_loss.item(), d_loss_frame.item(), g_loss.item(),
                D_solver.state_dict()['param_groups'][0]["lr"]))

        
        # calculate the fid score
        fid_image_path = os.path.join(args.output_dir, args.fid_fake_foldername, "images")
        if not os.path.exists(fid_image_path):
            os.makedirs(fid_image_path)
        vutils.save_image_forFID(fake, '{}/fake_samples_epoch_{}_{}.png'.format(fid_image_path, epoch+1, i), 
            normalize=True, pad_value=1, input_channels=args.input_channels, 
            imageSize=args.imageSize, fid_image_path=fid_image_path)

        # path_fid_images = [args.fid_real_path, fid_image_path]
        path_fid_images = [args.fid_real_path, os.path.join(args.output_dir, args.fid_fake_foldername)]
        print('calculate the fid score ...')
        # fid_value = fid_score.calculate_fid_score(path=path_fid_images,
        #     batch_size=args.batch_size, gpu=str(args.gpu+args.gpu_num-1))
        try:
            fid_value = fid_score.calculate_fid_score(path=path_fid_images,
                batch_size=args.batch_size, gpu=str(args.gpu))
        except:
            fid_value = best_fid
        if fid_value < best_fid:
            best_fid = fid_value
            best_epoch = epoch
            pickle_save_path = os.path.join(args.output_dir, args.pickle_dir)
            if not os.path.exists(pickle_save_path):
                os.makedirs(pickle_save_path)
            torch.save(G.state_dict(), '{}/G_best.pth'.format(pickle_save_path))
            torch.save(D.state_dict(), '{}/D_best.pth'.format(pickle_save_path))
            torch.save(D_frame_motion.state_dict(), '{}/D_frame_motion_best.pth'.format(pickle_save_path))

        print("\033[1;31m current_epoch[{}] current_fid[{}] \033[0m \033[1;34m best_epoch[{}] best_fid[{}] \033[0m".format(
            epoch, fid_value, best_epoch, best_fid))

        # save fid
        with open(os.path.join(args.output_dir, 'log_fid.txt'), 'a') as f:
            f.write("current_epoch[{}] current_fid[{}] best_epoch[{}] best_fid[{}] \n".format(
                epoch, fid_value, best_epoch, best_fid))


        # save images and sentence
        if (epoch + 1) % args.image_save_step == 0:
            image_path = os.path.join(args.output_dir, args.image_dir)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            vutils.save_image(fake, '{}/fake_samples_epoch_{}_{}.png'.format(image_path, epoch+1, i), 
                normalize=True, pad_value=1, input_channels=args.input_channels, imageSize=args.imageSize)
            # with open('{}/{:0>4d}.txt'.format(image_path, epoch+1), 'a') as f:
            #     for s in range(len(sentences)):
            #         f.write(sentences[s])
            #         f.write('\n')
            with open('{}/{:0>4d}.txt'.format(image_path, epoch+1), 'a') as f:
                for s in xrange(len(captions)):
                    for w in xrange(len(captions[s])):
                        idx = captions[s][w].item()
                        if idx==0:
                            break
                        word =  dataset.ixtoword[idx]
                        f.write(word)
                        f.write(' ')
                    f.write('\n')
        # # print the parameters of model
        # params=G.state_dict()
        # for k, v in params.items():
        #     print(k)
        # print(params['module.deconv.0.weight'])
        # assert False

        # checkpoint
        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = os.path.join(args.output_dir, args.pickle_dir)
            if not os.path.exists(pickle_save_path):
                os.makedirs(pickle_save_path)
            torch.save(G.state_dict(), '{}/G_epoch_{}.pth'.format(pickle_save_path, epoch+1))
            torch.save(D.state_dict(), '{}/D_epoch_{}.pth'.format(pickle_save_path, epoch+1))
            torch.save(D_frame_motion.state_dict(), '{}/D_frame_motion_epoch_{}.pth'.format(pickle_save_path, epoch+1))




