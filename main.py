
import argparse
from train import train
# from test import test
from test_forFID import test


def main(args):
    if args.test == False:
        train(args)
    else:
        test(args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ============================= Model Parmeters ============================= #
    parser.add_argument('--dataroot', type=str, default="/home/chenqi/Desktop/text2video/to_create_what_you_tell/data/moving_mnist_new_speeder_DAMSM", help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=48, help='the height / width of the input image to network')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='max epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='each batch size')
    parser.add_argument('--g_lr', type=float, default=0.0002,
                        help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                        help='discriminator learning rate')
    # parser.add_argument('--beta', type=tuple, default=(0.5, 0.999),
    #                     help='beta for adam')
    parser.add_argument('--beta', type=tuple, default=(0.9, 0.999),
                        help='beta for adam')
    parser.add_argument('--d_thresh', type=float, default=0.8,
                        help='for balance dsicriminator and generator')
    parser.add_argument('--z_size', type=int, default=100,
                        help='latent space size')
    parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
                        help='uniform: uni, normal: norm')
    # parser.add_argument('--bias', type=str2bool, default=True,
    #                     help='using cnn bias')
    parser.add_argument('--leak_value', type=float, default=0.2,
                        help='leakeay relu')
    # parser.add_argument('--cube_len', type=int, default=32,
    #                     help='cube length')
    # parser.add_argument('--obj', type=str, default="chair",
    #                     help='tranining dataset object category')
    parser.add_argument('--soft_label', type=str2bool, default=False,
                        help='using soft_label')
    parser.add_argument('--frame_num', type=int, default=16,
                        help='number of frame in a video')

    # ============================= Loss Parameters ============================= #
    parser.add_argument('--cls', type=str2bool, default=True,
                        help='using the mismatched sample in loss function')
    # parser.add_argument('--C', type=str2bool, default=False,
    #                     help='using the temporal coherence constraint loss')
    parser.add_argument('--A', type=str2bool, default=True,
                        help='using the temporal coherence adversarial loss')

    # ============================= Device Parameters ============================= #
    # parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enables cuda')
    parser.add_argument('--gpu', type=int, default=1, help='id of GPUs to use')
    parser.add_argument('--gpu_num', type=int, default=1, help='the number of gpu to use')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')

    # ============================= Dir Parameters ============================= #
    parser.add_argument('--output_dir', type=str, default="./output",
                        help='output path')
    # parser.add_argument('--input_dir', type=str, default='../input',
    #                     help='input path')
    parser.add_argument('--pickle_dir', type=str, default='/pickle/',
                        help='input path')
    # parser.add_argument('--log_dir', type=str, default='/log/',
    #                     help='for tensorboard log path save in output_dir + log_dir')
    parser.add_argument('--image_dir', type=str, default='/image/',
                        help='for output image path save in output_dir + image_dir')
    # parser.add_argument('--data_dir', type=str, default='/chair/',
    #                     help='dataset load path')

    # ============================= Step Parameter ============================= #
    parser.add_argument('--pickle_step', type=int, default=10,
                        help='pickle save at pickle_step epoch')
    # parser.add_argument('--log_step', type=int, default=1,
    #                     help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=1,
                        help='output image save at image_save_step epoch')

    # ============================= Text Parameter ============================= #
    parser.add_argument('--encoder', default='./Seq2Seq_with_attention/checkpoints_compression/encoder_75000.pth', help="path to sentence encoder")
    parser.add_argument('--hidden_size', type=int, default=256, help='the hidden size of encoder and decoder model')
    parser.add_argument('--file_path', default='./Seq2Seq_with_attention/data/train_text_collection_compression_speeder.txt', help="the path of the data file")
    parser.add_argument('--text_path', type=str, default='./data/moving_mnist_new_compression_speeder/train_text',
                        help='the path of image captions')
    parser.add_argument('--max_length', type=int, default=10, help='the max length of tokens in a sentence')

    # ============================= Image Parameter ============================= #
    parser.add_argument('--input_channels', type=int, default=1, help='the number of channel of input images (1 or 3)')
    parser.add_argument('--image_type', type=str, default="jpg", choices=["jpg", "png"], help='the type of the input images')
    parser.add_argument('--fid_fake_foldername', type=str, default="fid_images", help='the folder name to save the fid images')
    parser.add_argument('--fid_real_path', type=str, default="/home/chenqi/dataset/text2video/MSVD_DAMSM/train_image", help='the real path to save the real images to fid')


    # ============================= Other Parameter ============================= #
    # parser.add_argument('--simpleEmb', type=str2bool, default=False,
    #                     help='using the one-hot label as the conditional embedding')
    parser.add_argument('--init', type=str2bool, default=True,
                        help='using the initialization method')
    # parser.add_argument('--manualSeed', type=int, default=10, help='manual seed')
    parser.add_argument('--video_loss', type=str2bool, default=True,
                        help='using the video loss (D)')
    parser.add_argument('--frame_motion_loss', type=str2bool, default=True,
                        help='using the frame and motion loss (D)')

    # ============================= Train or Test ============================= #
    parser.add_argument('--test', type=str2bool, default=False,
                        help='only testing the model')

    # ============================= Checkpoints ============================= #
    parser.add_argument('--checkpoint_G', default='', help="path to checkpoint model of G")
    parser.add_argument('--checkpoint_D', default='', help="path to checkpoint model of D")
    parser.add_argument('--checkpoint_frame_motion_D', default='', help="path to checkpoint model of frame and motion")

    parser.add_argument('--checkpoint_text_encoder', default='', help="path to checkpoint model of text encoder")
    parser.add_argument('--checkpoint_image_encoder', default='', help="path to checkpoint model of image encoder")

    # ============================= Hyper-parameters ============================= #
    parser.add_argument('--lamb', type=float, default=1.0, help='TRAIN.SMOOTH.LAMBDA')
    # parser.add_argument('--norm_D', type=str, default="bn", choices=["bn", "sn"], help='normalization method of D')


    # # other parameters
    # parser.add_argument('--model_name', type=str, default="V2",
    #                     help='this model name for save pickle, logs, output image path and if model_name contain V2 modelV2 excute')
    # parser.add_argument('--use_tensorboard', type=str2bool, default=True,
    #                     help='using tensorboard logging')
    # parser.add_argument('--test_iter', type=int, default=10,
    #                     help='test_epoch number')
    # parser.add_argument('--test', type=str2bool, default=False,
    #                     help='for test')


    args = parser.parse_args()

    main(args)








