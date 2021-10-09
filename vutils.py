import torch
import math
import os
# import numpy as np
irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=16, padding=2,
               normalize=False, range=None, scale_each=False,
               pad_value=0, input_channels=3, imageSize=64):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if input_channels==1:
        # L (grey)
        # tensor = tensor.view(-1, 1, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 16, imageSize, imageSize)
    elif input_channels==3:
        # RGB
        # tensor = tensor.view(-1, 3, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 48, imageSize, imageSize)
    else:
        assert False

    # only one channel or not
    if input_channels==1:
        # tensor_new = torch.unsqueeze(tensor[0, 0, :, :], 0)
        tensor_new = tensor[0, 0, :, :].unsqueeze(0).unsqueeze(1)
    elif input_channels==3:
        tensor_new = torch.unsqueeze(tensor[0, 0:3, :, :], 0)
    else:
        assert False

    # for x in range(tensor.size(0)):
    for x in irange(0, 8):
        if x==0:
            # handle the first 16-frames
            for i in irange(1, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False
        else:
            # the last 7 16-frames
            for i in irange(0, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False


    from PIL import Image
    grid = make_grid(tensor_new, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    im = Image.fromarray(ndarr)
    im.save(filename)


def make_grid_forFID(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # if tensor.size(0) == 1:
    #     return tensor.squeeze()

    # # make the mini-batch of images into a grid
    # nmaps = tensor.size(0)
    # xmaps = min(nrow, nmaps)
    # ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    # grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    # k = 0
    # for y in irange(ymaps):
    #     for x in irange(xmaps):
    #         if k >= nmaps:
    #             break
    #         grid.narrow(1, y * height + padding, height - padding)\
    #             .narrow(2, x * width + padding, width - padding)\
    #             .copy_(tensor[k])
    #         k = k + 1
    # return grid

    return tensor


def save_image_forFID(tensor, filename, nrow=16, padding=2,
               normalize=False, range=None, scale_each=False,
               pad_value=0, input_channels=3, imageSize=64, fid_image_path=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if input_channels==1:
        # L (grey)
        # tensor = tensor.view(-1, 1, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 16, imageSize, imageSize)
    elif input_channels==3:
        # RGB
        # tensor = tensor.view(-1, 3, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 48, imageSize, imageSize)
    else:
        assert False

    # only one channel or not
    if input_channels==1:
        # tensor_new = torch.unsqueeze(tensor[0, 0, :, :], 0)
        tensor_new = tensor[0, 0, :, :].unsqueeze(0).unsqueeze(1)
    elif input_channels==3:
        tensor_new = torch.unsqueeze(tensor[0, 0:3, :, :], 0)
    else:
        assert False

    # for x in range(tensor.size(0)):
    # for x in irange(0, 8):
    for x in irange(0, tensor.size(0)):
        if x==0:
            # handle the first 16-frames
            for i in irange(1, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False
        else:
            # the last 7 16-frames
            for i in irange(0, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False


    from PIL import Image
    tensor = make_grid_forFID(tensor_new, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # print(tensor.shape)
    # assert False

    for i in xrange(tensor.size(0)):
        im = Image.fromarray(tensor[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        # im.save('{}_{:0>3d}.png'.format(filename[:-4], i))
        if fid_image_path!=None:
            im.save(os.path.join(fid_image_path, '{:0>4d}.png'.format(i)))
        else:
            im.save('{}_{:0>3d}.png'.format(filename[:-4], i))



def save_image_forfinalFID(tensor, filename, nrow=16, padding=2,
               normalize=False, range=None, scale_each=False,
               pad_value=0, input_channels=3, imageSize=64, 
               fid_image_path=None, counter_folder=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if input_channels==1:
        # L (grey)
        # tensor = tensor.view(-1, 1, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 16, imageSize, imageSize)
    elif input_channels==3:
        # RGB
        # tensor = tensor.view(-1, 3, 16, imageSize, imageSize)
        tensor = tensor.view(-1, 48, imageSize, imageSize)
    else:
        assert False

    # only one channel or not
    if input_channels==1:
        # tensor_new = torch.unsqueeze(tensor[0, 0, :, :], 0)
        tensor_new = tensor[0, 0, :, :].unsqueeze(0).unsqueeze(1)
    elif input_channels==3:
        tensor_new = torch.unsqueeze(tensor[0, 0:3, :, :], 0)
    else:
        assert False

    # for x in range(tensor.size(0)):
    # for x in irange(0, 8):
    for x in irange(0, tensor.size(0)):
        if x==0:
            # handle the first 16-frames
            for i in irange(1, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False
        else:
            # the last 7 16-frames
            for i in irange(0, 16):
                if input_channels==1:
                    tensor_new = torch.cat([tensor_new, tensor[x, i, :, :].unsqueeze(0).unsqueeze(1)], 0)
                elif input_channels==3:
                    tensor_new = torch.cat([tensor_new, torch.unsqueeze(tensor[x, i*3:(i+1)*3, :, :], 0)], 0)
                else:
                    assert False


    from PIL import Image
    tensor = make_grid_forFID(tensor_new, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # print(tensor.shape)
    # assert False

    counter_file = 0
    # current_folder = os.path.join(fid_image_path, '{:0>5d}'.format(counter_folder))
    current_folder = None
    for i in xrange(tensor.size(0)):
        if i%16==0:
            current_folder = os.path.join(fid_image_path, '{:0>5d}'.format(counter_folder))
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)
            counter_folder += 1
            counter_file = 0

        im = Image.fromarray(tensor[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        # im.save('{}_{:0>3d}.png'.format(filename[:-4], i))
        im.save(os.path.join(current_folder, '{:0>2d}.png'.format(counter_file)))
        counter_file += 1

    return counter_folder

