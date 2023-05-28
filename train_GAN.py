# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

from config        import cfg
from config        import update_config
from core.loss     import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils   import get_optimizer
from utils.utils   import save_checkpoint
from utils.utils   import create_logger
from utils.utils   import get_model_summary

import dataset
import models

from models.omnipose   import get_omnipose
from models.pose_hrnet import get_pose_net
from discriminator.discriminator import Discriminator
from torchvision.transforms.functional import resize 
import warnings
warnings.filterwarnings("ignore") 
from tqdm import tqdm 


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',          help='experiment configure file name',
                        default='experiments/coco/omnipose_w48_128x96_edouard.yaml', type=str)
    parser.add_argument('--opts',         help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',     help='model directory', type=str, default='')
    parser.add_argument('--logDir',       help='log directory', type=str, default='')
    parser.add_argument('--dataDir',      help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args


def main(args):
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    print('Model will be saved at: ',final_output_dir)
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_hrnet':
        model = get_pose_net(cfg, is_train=True)
    elif cfg.MODEL.NAME == 'omnipose':
        model = get_omnipose(cfg, is_train=True)

    discriminator = Discriminator(cfg.MODEL.NUM_JOINTS + 3, num_channels=4, num_joints=cfg.MODEL.NUM_JOINTS, num_residuals=3)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,}

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    # logger.info(get_model_summary(model, dump_input))

    if torch.cuda.is_available():
        model = model.cuda() 
        discriminator = discriminator.cuda()   

    # Define loss function and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([transforms.ToTensor(), normalize,]))
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize, ]) )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY)

    best_perf    = 0.0
    best_perf_01 = 0.0
    best_model   = False
    last_epoch   = -1
    
    optimizer_generator   = get_optimizer(cfg, model)
    optimizer_discriminator   = get_optimizer(cfg, discriminator)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        print('Loading checkpoint with accuracy of ', checkpoint['perf'], 'at epoch ',checkpoint['epoch'])
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        model_state_dict = model.state_dict()
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in checkpoint['state_dict'] and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict, strict=False)

        discriminator_dict = discriminator.state_dict()
        new_discriminator_dict = {}
        for k in discriminator_dict:
            if k in checkpoint['discriminator_state_dict'] and discriminator_dict[k].size() == checkpoint['discriminator_state_dict'][k].size():
                new_discriminator_dict[k] = checkpoint['discriminator_state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_discriminator_dict, strict=False)

        print('begin_epoch', begin_epoch)
        print('best_perf', best_perf)
        print('last_epoch',last_epoch)

        optimizer_generator.load_state_dict(checkpoint['optimizer'])
        optimizer_discriminator.load_state_dict(checkpoint['discriminator_optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_generator, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )

    for i in range(last_epoch):
        lr_scheduler.step()

    # In case you want to freeze layers prior to WASPv2, uncomment below:
    # model.requires_grad = False
    # model.waspv2.requires_grad = True

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        print(final_output_dir)

        # train for one epoch
        train_GAN(cfg, train_loader, 
                  model, 
                  discriminator, 
                  optimizer_discriminator, 
                  optimizer_generator,  
                  epoch,
                  final_output_dir, 
                  tb_log_dir)

        # evaluate on validation set

        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion,
            final_output_dir, tb_log_dir, writer_dict)
        perf_indicator_01 = 0
            
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_perf_01 = perf_indicator_01
            best_model = True

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer_generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'discriminator_optimizer': optimizer_discriminator.state_dict()
            }, best_model, final_output_dir)

        else:
            best_model = False

        print("Best so far: PCKh@0.5 = "+str(best_perf)+", PCKh@0.1 = "+str(best_perf_01))

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)



def _get_loss_disc(disc_output:torch.Tensor, real:bool, eps=1e-5):
    '''
    Get discriminator loss
    '''
    if real:
        loss = -torch.log(eps + disc_output).mean()
    else:
        loss = -torch.log(eps + 1 - disc_output).mean()
    return loss

def get_loss_disc(disc_fake:torch.Tensor, disc_real:torch.Tensor) -> float:

    loss = _get_loss_disc(disc_fake, real=False) + _get_loss_disc(disc_real, real=True)
    return loss


def get_loss_gen(outputs:torch.Tensor, disc_fake:torch.Tensor, target:torch.Tensor) -> float:

    loss_gen = ((outputs - target)**2).mean()
    loss_disc = _get_loss_disc(disc_output=disc_fake, real=True)
    loss = loss_gen + loss_disc
    return loss


def _resize_images_batch(images:torch.Tensor, dest_size:tuple) -> torch.Tensor:
    res = []
    batch_size = images.shape[0]
    for i in range(batch_size):
        cur_image = images[i]
        res.append(resize(img=cur_image, size=dest_size))

    return torch.stack(res, dim=0)



def train_GAN(cfg, train_loader, 
                  model, 
                  discriminator, 
                  optimizer_discriminator, 
                  optimizer_generator,  
                  epoch,
                  final_output_dir, 
                  tb_log_dir):
    
    tbar = tqdm(train_loader)

    print("Epoch ",str(epoch),":")
    model.train()
    discriminator.train()

    avg_disc_loss = 0
    avg_gen_loss = 0

    for i, (input, target, target_weight, meta) in enumerate(tbar):

        # input_resized = _resize_images_batch(input, dest_size=(cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]))
        if torch.cuda.is_available():
            input  = input.cuda()
            # input_resized  = input_resized.cuda()
            target = target.cuda()
            target_weight = target_weight.cuda()


        # optimize discriminator
        optimizer_discriminator.zero_grad()
        outputs = model(input)
        disc_real = discriminator(torch.cat([target, input], axis=1))
        print("input shape:", input.shape, "target: ", target.shape, "outputs:", outputs.shape)
        disc_fake = discriminator(torch.cat([outputs, input], axis=1))
        loss_disc = get_loss_disc(disc_fake, disc_real)

        avg_disc_loss += loss_disc.item()
        loss_disc.backward()
        optimizer_discriminator.step()

        # optimize generator
        optimizer_generator.zero_grad()
        outputs = model(input)
        disc_fake = discriminator(torch.cat([outputs, input], axis=1))
        loss_gen = get_loss_gen(outputs, disc_fake, target)

        avg_gen_loss += loss_gen.item()
        loss_gen.backward()
        optimizer_generator.step()
    
    print(f"loss gen: {avg_gen_loss / i}, loss disc: {avg_disc_loss / i}")


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
