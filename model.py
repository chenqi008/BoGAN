import torch
import torch.nn as nn
# from torch.nn.utils import spectral_norm
from spectral_normalization import SpectralNorm as spectral_norm


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.ConvTranspose3d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class GLU(torch.nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.nn.functional.sigmoid(x[:, nc:])


class CA_NET(torch.nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, args):
        super(CA_NET, self).__init__()
        self.args = args
        self.t_dim = args.hidden_size
        self.c_dim = 100
        self.fc = torch.nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        # self.conv_context = conv1x1(cdf, idf)
        self.conv_context = nn.Conv2d(cdf, idf, 1, 1, 0, bias=False)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x channels x queryL x ih x iw (idf=ihxiwxchannels)
            context: batch x cdf x sourceL
        """
        cln, ih, iw = input.size(1), input.size(3), input.size(4)
        # queryL = ih * iw * cln
        idf = ih * iw * cln
        queryL = input.size(2)
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.permute(0, 2, 1, 3, 4).contiguous()
        targetT = target.view(batch_size, -1, idf)
        # targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        # weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        # attn = attn.view(batch_size, -1, ih, iw)
        weightedContext = weightedContext.view(batch_size, -1, cln, ih, iw)
        weightedContext = weightedContext.permute(0, 2, 1, 3, 4).contiguous()
        # attn = attn.view(batch_size, -1, cln, ih, iw)
        # attn = attn.permute(0, 2, 1, 3, 4).contiguous()

        return weightedContext, attn


# ========================== #
#  Contain the Linear layer  #
# ========================== #

class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.ca_net = CA_NET(self.args)

        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(self.args.z_size+self.args.hidden_size, 1024, bias=False),  # 64x(100+256)->64x1024
            torch.nn.Linear(self.args.z_size+256, 1024, bias=False),  # 64x(100+256)->64x1024
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512 * 2 * (self.args.imageSize // 8) * (self.args.imageSize // 8), bias=False),  # 64x1024->64x(512x2x8x8)
            torch.nn.BatchNorm1d(512 * 2 * (self.args.imageSize // 8) * (self.args.imageSize // 8)),
            torch.nn.ReLU(inplace=True),
        )
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),  # 64x512x2x8x8->64x256x4x16x16
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),  # 64x256x4x16x16->64x128x8x32x32
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),  # 64x128x8x32x32->64x64x16x64x64
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(inplace=True),
            # torch.nn.ConvTranspose3d(64, 1, 3, 1, 1, bias=False),  # 64x64x16x64x64->64x1x16x64x64
            torch.nn.ConvTranspose3d(64, args.input_channels, 3, 1, 1, bias=False),  # 64x64x16x64x64->64x1x16x64x64
            torch.nn.Tanh(),
        )
        if self.args.init:
            print("Initialize _G")
            initialize_weights(self)

    def forward(self, x, sent_emb):
        c_code, mu, logvar = self.ca_net(sent_emb)
        x_c_code = torch.cat((x, c_code), 1)
        # print(x_c_code.size())  # torch.Size([64, 100+256])
        out = self.fc(x_c_code)
        # print(out.size())  # torch.Size([64, (512x2x8x8)])
        out = out.view(-1, 512, 2, (self.args.imageSize // 8), (self.args.imageSize // 8))
        # print(out.size())  # torch.Size([64, 512, 2, 8, 8])
        out = self.deconv(out)
        # print(out.size())  # torch.Size([64, 1, 16, 64, 64])
        return out, mu, logvar


# # ============================= #
# # Not Contain the Linear layer  #
# # ============================= #

# class _G(torch.nn.Module):
#     def __init__(self, args):
#         super(_G, self).__init__()
#         self.args = args
#         self.ca_net = CA_NET(self.args)

#         self.deconv = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(self.args.z_size+100, 512, 
#                 (2, self.args.imageSize // 8, self.args.imageSize // 8), 1, 0, bias=False),
#             torch.nn.BatchNorm3d(512),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),  # 64x512x2x8x8->64x256x4x16x16
#             torch.nn.BatchNorm3d(256),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),  # 64x256x4x16x16->64x128x8x32x32
#             torch.nn.BatchNorm3d(128),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),  # 64x128x8x32x32->64x64x16x64x64
#             torch.nn.BatchNorm3d(64),
#             torch.nn.ReLU(inplace=True),
#             # torch.nn.ConvTranspose3d(64, 1, 3, 1, 1, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.ConvTranspose3d(64, args.input_channels, 1, 1, 0, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.Tanh(),
#         )
#         if self.args.init:
#             print("Initialize _G")
#             initialize_weights(self)

#     def forward(self, x, sent_emb):
#         c_code, mu, logvar = self.ca_net(sent_emb)
#         x_c_code = torch.cat((x, c_code), 1)
#         # print(x_c_code.size())  # torch.Size([64, 100+256])
#         x_c_code = x_c_code.view(-1, self.args.z_size+100, 1, 1, 1)
#         out = self.deconv(x_c_code)
#         # print(out.size())  # torch.Size([64, 1, 16, 64, 64])
#         return out, mu, logvar


# class _G(torch.nn.Module):
#     def __init__(self, args):
#         super(_G, self).__init__()
#         self.args = args
#         self.ca_net = CA_NET(self.args)

#         self.fc = torch.nn.Sequential(
#             # torch.nn.Linear(self.args.z_size+self.args.hidden_size, 1024, bias=False),  # 64x(100+256)->64x1024
#             torch.nn.Linear(self.args.z_size+100, 1024, bias=False),  # 64x(100+256)->64x1024
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(1024, 512 * 2 * (self.args.imageSize // 8) * (self.args.imageSize // 8), bias=False),  # 64x1024->64x(512x2x8x8)
#             torch.nn.BatchNorm1d(512 * 2 * (self.args.imageSize // 8) * (self.args.imageSize // 8)),
#             torch.nn.ReLU(inplace=True),
#         )
#         self.deconv = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),  # 64x512x2x8x8->64x256x4x16x16
#             torch.nn.BatchNorm3d(256),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),  # 64x256x4x16x16->64x128x8x32x32
#             torch.nn.BatchNorm3d(128),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),  # 64x128x8x32x32->64x64x16x64x64
#             torch.nn.BatchNorm3d(64),
#             torch.nn.ReLU(inplace=True),
#             # torch.nn.ConvTranspose3d(64, 1, 3, 1, 1, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.ConvTranspose3d(64, args.input_channels, 3, 1, 1, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.Tanh(),
#         )
#         # for attention
#         self.att = GlobalAttentionGeneral(args.input_channels*args.imageSize*args.imageSize, args.hidden_size)

#         self.deconv2 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(args.input_channels*2, args.input_channels, 1, 1, 0, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.BatchNorm3d(args.input_channels),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.ConvTranspose3d(args.input_channels, args.input_channels, 1, 1, 0, bias=False),  # 64x64x16x64x64->64x1x16x64x64
#             torch.nn.Tanh(),
#             )

#         if self.args.init:
#             print("Initialize _G")
#             initialize_weights(self)

#     def forward(self, z_code, sent_emb, word_embs, mask):
#         c_code, mu, logvar = self.ca_net(sent_emb)
#         z_c_code = torch.cat((z_code, c_code), 1)
#         # print(x_c_code.size())  # torch.Size([64, 100+256])
#         out = self.fc(z_c_code)
#         # print(out.size())  # torch.Size([64, (512x2x8x8)])
#         out = out.view(-1, 512, 2, (self.args.imageSize // 8), (self.args.imageSize // 8))
#         # print(out.size())  # torch.Size([64, 512, 2, 8, 8])
#         out = self.deconv(out)
#         # print(out.size())  # torch.Size([64, 1, 16, 64, 64])
#         # out = out.view(-1, self.args.input_channels, self.args.imageSize, self.args.imageSize)
#         # # print(out.size())  # torch.Size([64*16, 1, 64, 64])

#         self.att.applyMask(mask)
#         out2, att = self.att(out, word_embs)
#         out1_2 = torch.cat((out, out2), 1)

#         out1_2 = self.deconv2(out1_2)

#         return out1_2, mu, logvar


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        # if self.args.norm_D=='bn':
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv3d(1, 64, 4, 2, 1, bias=False),  # 64x1x16x64x64->64x64x8x32x32
            torch.nn.Conv3d(args.input_channels, 64, 4, 2, 1, bias=False),  # 64x1x16x64x64->64x64x8x32x32
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv3d(64, 128, 4, 2, 1, bias=False),  # 64x64x8x32x32->64x128x4x16x16
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv3d(128, 256, 4, 2, 1, bias=False),  # 64x128x4x16x16->64x256x2x8x8
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv3d(256, 512, 4, 2, 1, bias=False),  # 64x256x2x8x8->64x512x1x4x4
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False),  # 64x(512+256)x1x4x4->64x512x1x4x4
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Linear(1024, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        # elif self.args.norm_D=='sn':
        #     self.conv1 = torch.nn.Sequential(
        #         # torch.nn.Conv3d(1, 64, 4, 2, 1, bias=False),  # 64x1x16x64x64->64x64x8x32x32
        #         spectral_norm(torch.nn.Conv3d(args.input_channels, 64, 4, 2, 1, bias=False)),  # 64x1x16x64x64->64x64x8x32x32
        #         # spectral_norm(64),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv3d(64, 128, 4, 2, 1, bias=False)),  # 64x64x8x32x32->64x128x4x16x16
        #         # spectral_norm(128),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv3d(128, 256, 4, 2, 1, bias=False)),  # 64x128x4x16x16->64x256x2x8x8
        #         # spectral_norm(256),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv3d(256, 512, 4, 2, 1, bias=False)),  # 64x256x2x8x8->64x512x1x4x4
        #         # spectral_norm(512),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #     )
        #     self.conv2 = torch.nn.Sequential(
        #         spectral_norm(torch.nn.Conv3d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False)),  # 64x(512+256)x1x4x4->64x512x1x4x4
        #         # spectral_norm(512),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #     )
        #     self.fc = torch.nn.Sequential(
        #         spectral_norm(torch.nn.Linear(512 * 1 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False)),
        #         # spectral_norm(1024),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         torch.nn.Linear(1024, 1, bias=False),
        #         torch.nn.Sigmoid(),
        #     )

        if self.args.init:
            print("Initialize _D")
            initialize_weights(self)

    def forward(self, x, embedding):
        # print(x.size())  # torch.Size([64, 16, 64, 64])
        # print(embedding.size())  # torch.Size([64, 256])
        # out = x.view(-1, 1, 16, self.args.imageSize, self.args.imageSize)
        out = x.view(-1, self.args.input_channels, 16, self.args.imageSize, self.args.imageSize)
        # print(out.size())  # torch.Size([64, 1, 16, 64, 64])
        out = self.conv1(out)
        # print(out.size())  # torch.Size([64, 512, 1, 4, 4])
        embedding = embedding.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(
            embedding.size()[0], embedding.size()[1], 1, out.size()[3], out.size()[4])
        # print(embedding.size())  # torch.Size([64, 256]) -> torch.Size([64, 256, 1, 4, 4])
        out = torch.cat((out, embedding), 1)
        # print(out.size())  # torch.Size([64, 768, 1, 4, 4])
        out = self.conv2(out)
        # print(out.size())  # torch.Size([64, 512, 1, 4, 4])
        out = out.view(-1, 512 * 1 * (self.args.imageSize // 16) * (self.args.imageSize // 16))
        # print(out.size())  # torch.Size([64, 512x1x4x4])
        out = self.fc(out)
        # print(out.size())  # torch.Size([64, 1])
        out = out.squeeze()
        return out


class _D_frame_motion(torch.nn.Module):
    def __init__(self, args):
        super(_D_frame_motion, self).__init__()
        self.args = args
        # if self.args.norm_D=='bn':
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # (64x16)x1x64x64->(64x16)x64x32x32
            torch.nn.Conv2d(args.input_channels, 64, 4, 2, 1, bias=False),  # (64x16)x1x64x64->(64x16)x64x32x32
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (64x16)x64x32x32->(64x16)x128x16x16
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (64x16)x128x16x16->(64x16)x256x8x8
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (64x16)x256x8x8->(64x16)x512x4x4
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        )
        self.conv_frame = torch.nn.Sequential(
            torch.nn.Conv2d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False),  # (64x16)x(512+256)x4x4->(64x16)x512x4x4
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        )
        self.fc_frame = torch.nn.Sequential(
            torch.nn.Linear(512 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            torch.nn.Linear(1024, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        if self.args.A:
            self.conv_motion = torch.nn.Sequential(
                torch.nn.Conv2d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False),  # (64x16)x(512+256)x4x4->(64x16)x512x4x4
                torch.nn.BatchNorm2d(512),
                torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
            )
            self.fc_motion = torch.nn.Sequential(
                torch.nn.Linear(512 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False),
                torch.nn.BatchNorm1d(1024),
                torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
                torch.nn.Linear(1024, 1, bias=False),
                torch.nn.Sigmoid(),
            )
        # elif self.args.norm_D=='sn':
        #     self.conv1 = torch.nn.Sequential(
        #         # torch.nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # (64x16)x1x64x64->(64x16)x64x32x32
        #         spectral_norm(torch.nn.Conv2d(args.input_channels, 64, 4, 2, 1, bias=False)),  # (64x16)x1x64x64->(64x16)x64x32x32
        #         # spectral_norm(64),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False)),  # (64x16)x64x32x32->(64x16)x128x16x16
        #         # spectral_norm(128),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False)),  # (64x16)x128x16x16->(64x16)x256x8x8
        #         # spectral_norm(256),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         spectral_norm(torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False)),  # (64x16)x256x8x8->(64x16)x512x4x4
        #         # spectral_norm(512),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #     )
        #     self.conv_frame = torch.nn.Sequential(
        #         spectral_norm(torch.nn.Conv2d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False)),  # (64x16)x(512+256)x4x4->(64x16)x512x4x4
        #         # spectral_norm(512),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #     )
        #     self.fc_frame = torch.nn.Sequential(
        #         spectral_norm(torch.nn.Linear(512 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False)),
        #         # spectral_norm(1024),
        #         torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         torch.nn.Linear(1024, 1, bias=False),
        #         torch.nn.Sigmoid(),
        #     )
        #     if self.args.A:
        #         self.conv_motion = torch.nn.Sequential(
        #             spectral_norm(torch.nn.Conv2d(512+self.args.hidden_size, 512, 3, 1, 1, bias=False)),  # (64x16)x(512+256)x4x4->(64x16)x512x4x4
        #             # spectral_norm(512),
        #             torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #         )
        #         self.fc_motion = torch.nn.Sequential(
        #             spectral_norm(torch.nn.Linear(512 * (self.args.imageSize // 16) * (self.args.imageSize // 16), 1024, bias=False)),
        #             # spectral_norm(1024),
        #             torch.nn.LeakyReLU(self.args.leak_value, inplace=True),
        #             torch.nn.Linear(1024, 1, bias=False),
        #             torch.nn.Sigmoid(),
        #         )

        if self.args.init:
            print("Initialize _D_frame_motion")
            initialize_weights(self)

    def forward(self, x, embedding):
        # print(x.size())  # torch.Size([64, 16, 64, 64])
        # print(embedding.size())  # torch.Size([64, 256])
        # out = x.view(-1, 1, self.args.imageSize, self.args.imageSize)
        out = x.view(-1, self.args.input_channels, self.args.imageSize, self.args.imageSize)
        # print(out.size()) # torch.Size([64x16=1024, 1, 64, 64])
        out = self.conv1(out)
        # print(out.size()) # torch.Size([64x16=1024, 512, 4, 4])

        if self.args.A:
            out_motion = out.view(x.size()[0], self.args.frame_num, out.size()[1], \
                out.size()[2], out.size()[3])
            # print(out_motion.size()) # torch.Size([64, 16, 512, 4, 4])
            out_motion_pre = out_motion[:, :-1, :, :, :].clone()
            out_motion_post = out_motion[:, 1:, :, :, :].clone()
            # out_motion = (out_motion_post - out_motion_pre)**2
            out_motion = out_motion_post - out_motion_pre
            # print(out_motion.size()) # torch.Size([64, 15, 512, 4, 4])
            out_motion = out_motion.view(x.size()[0]*(self.args.frame_num-1), \
                out.size()[1], out.size()[2], out.size()[3])
            # print(out_motion.size()) # torch.Size([64x15=960, 512, 4, 4])

        embedding_frame = embedding.repeat(1, out.size()[0]/embedding.size()[0]).view(out.size()[0], -1)
        # print(embedding_frame.size())  # torch.Size([64, 256]) -> torch.Size([1024, 256])
        embedding_frame = embedding_frame.unsqueeze(2).unsqueeze(3).expand(
            embedding_frame.size()[0], embedding_frame.size()[1], out.size()[2], out.size()[3])
        # print(embedding_frame.size())  # torch.Size([1024, 256]) -> torch.Size([1024, 256, 4, 4])
        out = torch.cat((out, embedding_frame), 1)
        # print(out.size())  # torch.Size([1024, 768, 4, 4])
        out = self.conv_frame(out)
        # print(out.size()) # torch.Size([64x16=1024, 512, 4, 4])
        out = out.view(-1, 512 * (self.args.imageSize // 16) * (self.args.imageSize // 16))
        # print(out.size())  # torch.Size([1024, 512x4x4])
        out = self.fc_frame(out)
        # print(out.size())  # torch.Size([1024, 1])
        out = out.squeeze()

        if self.args.A:
            embedding_motion = embedding.repeat(1, out_motion.size()[0]/embedding.size()[0]).view(out_motion.size()[0], -1)
            # print(embedding_motion.size())  # torch.Size([64, 256]) -> torch.Size([960, 256])
            embedding_motion = embedding_motion.unsqueeze(2).unsqueeze(3).expand(
                embedding_motion.size()[0], embedding_motion.size()[1], out_motion.size()[2], out_motion.size()[3])
            # print(embedding_motion.size())  # torch.Size([960, 256]) -> torch.Size([960, 256, 4, 4])
            out_motion = torch.cat((out_motion, embedding_motion), 1)
            # print(out_motion.size())  # torch.Size([960, 768, 4, 4])
            out_motion = self.conv_motion(out_motion)
            # print(out_motion.size()) # torch.Size([64x15=960, 512, 4, 4])
            out_motion = out_motion.view(-1, 512 * (self.args.imageSize // 16) * (self.args.imageSize // 16))
            # print(out_motion.size())  # torch.Size([960, 512x4x4])
            out_motion = self.fc_motion(out_motion)
            # print(out.size())  # torch.Size([960, 1])
            out_motion = out_motion.squeeze()
            return out, out_motion

        return out





